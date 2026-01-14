#!/usr/bin/env python3
"""
Active Learning Runner for Label Studio

This script provides an interactive active learning workflow that:
1. Loads a machine learning model
2. Identifies the most uncertain samples
3. Prompts user to annotate them via Label Studio API
4. Retrains the model with new annotations
5. Repeats the process

Usage:
    python active_learning_runner.py --config config.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Third-party imports
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, IntPrompt, Prompt
    from rich.table import Table
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install with: pip install rich")
    sys.exit(1)

from .active_learning import ActiveLearningManager, WebhookHandler
from .base import APIClient, Config


class ActiveLearningRunner:
    """Main runner for active learning workflows."""

    def __init__(self, config_path: str = "config.json"):
        self.console = Console()
        self.config = None
        self.client = None
        self.project_id = None
        self.al_manager = None

        self.load_config(config_path)

    def load_config(self, config_path: str):
        """Load configuration and initialize API client."""
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            self.config = Config(**config_data)
            self.client = APIClient(self.config)

            if not self.client.test_connection():
                self.console.print("[red]Failed to connect to Label Studio API[/red]")
                sys.exit(1)

            self.console.print("[green]Connected to Label Studio API[/green]")

        except FileNotFoundError:
            self.console.print(f"[red]Config file not found: {config_path}[/red]")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"[red]Failed to load config: {e}[/red]")
            sys.exit(1)

    def select_project(self) -> Optional[int]:
        """Interactive project selection."""
        projects = self.client.get_projects()

        if not projects or not projects.get("results"):
            self.console.print("[red]No projects found[/red]")
            return None

        self.console.print("\n[bold blue]Available Projects:[/bold blue]")

        table = Table()
        table.add_column("Index", style="cyan")
        table.add_column("ID", style="magenta")
        table.add_column("Title", style="green")
        table.add_column("Tasks", style="yellow")

        for i, project in enumerate(projects["results"], 1):
            table.add_row(
                str(i),
                str(project.get("id")),
                project.get("title", "Untitled"),
                str(project.get("task_number", 0)),
            )

        self.console.print(table)

        while True:
            try:
                choice = IntPrompt.ask("Select project", default=1)
                if 1 <= choice <= len(projects["results"]):
                    selected_project = projects["results"][choice - 1]
                    project_id = selected_project["id"]
                    self.console.print(
                        f"[green]Selected project: {selected_project.get('title')} (ID: {project_id})[/green]"
                    )
                    return project_id
                else:
                    self.console.print("[red]Invalid selection[/red]")
            except (ValueError, KeyboardInterrupt):
                self.console.print("[yellow]Selection cancelled[/yellow]")
                return None

    def setup_active_learning(self, project_id: int):
        """Setup active learning configuration."""
        self.console.print(
            f"\n[bold blue]Setting up Active Learning for Project {project_id}[/bold blue]"
        )

        # Check if project has ML backends
        ml_backends = self.client.list_ml_backends(project_id)
        if not ml_backends:
            self.console.print(
                "[yellow]No ML backends found. Please add one first.[/yellow]"
            )
            return False

        self.console.print(f"[green]Found {len(ml_backends)} ML backend(s)[/green]")

        # Configure parameters
        batch_size = IntPrompt.ask("Batch size for active learning", default=10)

        uncertainty_strategies = ["entropy", "least_confidence", "margin", "random"]
        self.console.print("\n[bold]Uncertainty Sampling Strategies:[/bold]")
        for i, strategy in enumerate(uncertainty_strategies, 1):
            self.console.print(f"{i}. {strategy}")

        strategy_choice = IntPrompt.ask("Select uncertainty strategy", default=1)
        uncertainty_strategy = uncertainty_strategies[strategy_choice - 1]

        min_confidence = float(
            Prompt.ask("Minimum uncertainty threshold", default="0.3")
        )

        # Initialize active learning manager
        self.al_manager = ActiveLearningManager(
            api_client=self.client,
            project_id=project_id,
            uncertainty_strategy=uncertainty_strategy,
            diversity_strategy="cluster",
            batch_size=batch_size,
            min_confidence_threshold=min_confidence,
        )

        self.project_id = project_id

        # Display configuration
        config_panel = f"""[green]Active Learning Configuration:[/green]
        
• Project ID: {project_id}
• Batch Size: {batch_size}
• Uncertainty Strategy: {uncertainty_strategy}
• Min Confidence Threshold: {min_confidence}
• ML Backends: {len(ml_backends)}"""

        self.console.print(
            Panel(config_panel, title="Configuration", border_style="blue")
        )
        return True

    def run_active_learning_loop(self):
        """Run the main active learning loop."""
        if not self.al_manager:
            self.console.print("[red]Active learning not configured[/red]")
            return

        self.console.print("\n[bold green]Starting Active Learning Loop[/bold green]")

        max_iterations = IntPrompt.ask("Maximum iterations", default=10)

        for iteration in range(max_iterations):
            self.console.print(
                f"\n[bold cyan]--- Iteration {iteration + 1} ---[/bold cyan]"
            )

            # Check if we should continue
            if iteration > 0:
                if not Confirm.ask("Continue to next iteration?", default=True):
                    break

            # Select next batch
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    "Selecting optimal batch for annotation...", total=None
                )

                selected_tasks = self.al_manager.select_next_batch()

                if not selected_tasks:
                    progress.update(
                        task, description="[yellow]No more tasks to annotate[/yellow]"
                    )
                    self.console.print(
                        "[yellow]No tasks available for annotation. Stopping.[/yellow]"
                    )
                    break

                progress.update(
                    task,
                    description=f"[green]Selected {len(selected_tasks)} tasks[/green]",
                )

            # Display selected tasks
            self.display_selected_tasks(selected_tasks)

            # Launch annotation interface
            if Confirm.ask(
                "Launch Label Studio interface for annotation?", default=True
            ):
                self.launch_annotation_interface(selected_tasks)

            # Wait for annotations
            self.console.print(
                "[yellow]Please annotate the selected tasks in Label Studio...[/yellow]"
            )
            input("Press Enter when annotation is complete...")

            # Update labeled tasks
            self.al_manager.update_labeled_tasks(selected_tasks)

            # Trigger retraining
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Triggering model retraining...", total=None)

                if self.al_manager.trigger_model_retraining():
                    progress.update(
                        task,
                        description="[green]Training triggered successfully[/green]",
                    )
                else:
                    progress.update(
                        task, description="[red]Failed to trigger training[/red]"
                    )

            # # Wait for training completion
            # if Confirm.ask("Wait for training to complete?", default=True):
            #     with Progress(
            #         SpinnerColumn(),
            #         TextColumn("[progress.description]{task.description}"),
            #         console=self.console,
            #     ) as progress:
            #         task = progress.add_task("Waiting for training completion...", total=None)
            #
            #         if self.al_manager.wait_for_training_completion():
            #             progress.update(task, description="[green]Training completed[/green]")
            #         else:
            #             progress.update(task, description="[yellow]Training timeout or failed[/yellow]")

            # Show current stats
            self.display_learning_stats()

        self.console.print("\n[bold green]Active Learning Loop Completed![/bold green]")

    def display_selected_tasks(self, task_ids: list):
        """Display the selected tasks for annotation."""
        table = Table(title="Selected Tasks for Active Learning")
        table.add_column("Task ID", style="cyan")
        table.add_column("Action Required", style="green")
        table.add_column("Priority", style="yellow")

        for task_id in task_ids:
            table.add_row(str(task_id), "Annotation needed", "High uncertainty")

        self.console.print(table)

    def launch_annotation_interface(self, task_ids: list):
        """Launch the Label Studio annotation interface."""
        import webbrowser

        # Create URL for the annotation interface
        base_url = self.config.api_base_url
        project_id = self.project_id

        # Create filtered view for selected tasks
        task_filter = ",".join(map(str, task_ids))
        url = f"{base_url}/projects/{project_id}/data?tab=0&task={task_filter}"

        self.console.print(f"[blue]Opening: {url}[/blue]")

        try:
            webbrowser.open(url)
            self.console.print(
                "[green]Label Studio interface opened in browser[/green]"
            )
        except Exception as e:
            self.console.print(f"[red]Failed to open browser: {e}[/red]")
            self.console.print(f"[yellow]Please manually open: {url}[/yellow]")

    def display_learning_stats(self):
        """Display current active learning statistics."""
        if not self.al_manager:
            return

        stats = self.al_manager.get_learning_stats()

        stats_table = Table(title="Active Learning Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")

        for key, value in stats.items():
            if key != "timestamp":
                stats_table.add_row(key.replace("_", " ").title(), str(value))

        self.console.print(stats_table)

    def run(self):
        """Main entry point for the active learning runner."""
        self.console.print(
            "[bold blue]🤖 Active Learning Runner for Label Studio[/bold blue]\n"
        )

        # Select project
        project_id = self.select_project()
        if not project_id:
            return

        # Setup active learning
        if not self.setup_active_learning(project_id):
            return

        # Run the main loop
        self.run_active_learning_loop()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Active Learning Runner for Label Studio"
    )
    parser.add_argument(
        "--config", default="config.json", help="Configuration file path"
    )

    args = parser.parse_args()

    runner = ActiveLearningRunner(args.config)
    runner.run()


if __name__ == "__main__":
    main()
