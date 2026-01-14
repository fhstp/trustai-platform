import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .base import APIClient, Config

# Third-party imports
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.syntax import Syntax
    from rich.table import Table
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install with: pip install tabulate click rich")
    sys.exit(1)


class TemplateManager:
    """Manages HTML templates for Label Studio frontend interfaces."""

    def __init__(self, template_dir: str = "frontend_labeling_templates"):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)

    def list_templates(self) -> List[str]:
        """List available HTML templates."""
        templates = []
        for file in self.template_dir.glob("*.html"):
            templates.append(file.stem)
        return templates

    def load_template(self, template_name: str) -> str:
        """Load HTML template content."""
        template_path = self.template_dir / f"{template_name}.html"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()


class baseHelper:
    """
    Interactive Label Studio Demo Application.
    """

    def __init__(self):
        """Initialize the demo application."""
        self.console = Console()
        self.client = None
        self.config = None
        self.current_project = None
        self.template_manager = TemplateManager()

    def load_config(self, config_path: str = "config.json") -> bool:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(config_path):
                self.console.print(f"[red]Config file not found: {config_path}[/red]")
                return self.create_config_file(config_path)

            with open(config_path, "r") as f:
                config_data = json.load(f)

            required_fields = ["api_base_url", "api_key"]
            for field in required_fields:
                if field not in config_data:
                    self.console.print(
                        f"[red]Missing required field in config: {field}[/red]"
                    )
                    return False

            self.config = Config(**config_data)
            self.client = APIClient(self.config)

            self.console.print(f"[green]Configuration loaded successfully[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]Failed to load config: {e}[/red]")
            return False

    def create_config_file(self, config_path: str) -> bool:
        """
        Create a new configuration file interactively.

        Args:
            config_path: Path where to create the config file

        Returns:
            True if successful, False otherwise
        """
        self.console.print("[yellow]Creating new configuration file...[/yellow]")

        api_base_url = Prompt.ask(
            "Enter Label Studio API base URL", default="http://localhost:8080"
        )
        api_key = Prompt.ask("Enter your API key", password=True)
        if not api_key or len(api_key.strip()) < 10:
            self.console.print(
                "[red]Invalid API key. Must be at least 10 characters.[/red]"
            )
            return False

        config_data = {
            "api_base_url": api_base_url,
            "api_key": api_key,
            "timeout": 30,
            "verify_ssl": True,
        }

        try:
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            self.console.print(f"[green]Configuration saved to {config_path}[/green]")
            return self.load_config(config_path)

        except Exception as e:
            self.console.print(f"[red]Failed to create config file: {e}[/red]")
            return False

    def test_connection(self):
        """Test connection to Label Studio API."""
        if not self.client:
            self.console.print("[red]No client initialized[/red]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Testing connection...", total=None)

            if self.client.test_connection():
                progress.update(
                    task, description="[green]Connection successful![/green]"
                )
                progress.stop()
                self.console.print("[green]Connected to Label Studio API[/green]")
            else:
                progress.update(task, description="[red]Connection failed![/red]")
                progress.stop()
                self.console.print("[red]Failed to connect to Label Studio API[/red]")
                quit()

    def list_projects(self):
        """List all projects in a formatted table."""
        if not self.client:
            self.console.print("[red]No client initialized[/red]")
            return

        projects = self.client.get_projects()

        if not projects:
            self.console.print("[yellow]No projects found[/yellow]")
            return

        table = Table(title="Label Studio Projects")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="magenta")
        table.add_column("Description", style="green")
        table.add_column("Tasks", style="yellow")
        table.add_column("Created", style="blue")

        for project in projects["results"]:
            created_at = project.get("created_at", "")
            if created_at:
                try:
                    created_at = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d %H:%M")
                except:
                    pass

            table.add_row(
                str(project.get("id", "")),
                project.get("title", ""),
                (
                    project.get("description", "")[:50] + "..."
                    if len(project.get("description", "")) > 50
                    else project.get("description", "")
                ),
                str(project.get("task_number", 0)),
                created_at,
            )

        self.console.print(table)

    def create_project_interactive(self):
        """Create a new project interactively."""
        if not self.client:
            self.console.print("[red]No client initialized[/red]")
            return

        self.console.print("[bold blue]Creating New Project[/bold blue]")

        title = Prompt.ask("Enter project title")
        description = Prompt.ask("Enter project description", default="")

        # Provide some common sampling configurations
        sampling_configs = {
            "1": "Sequential sampling",
            "2": "Uniform sampling",
            "3": "Uncertainty sampling (Active Learning)",
        }

        # Provide some common label configurations
        label_configs = {
            "1": {
                "name": "Text Classification",
                "config": """<View>
  <Text name="text" value="$text"/>
  <Choices name="sentiment" toName="text">
    <Choice value="positive"/>
    <Choice value="negative"/>
    <Choice value="neutral"/>
  </Choices>
</View>""",
            },
            "2": {
                "name": "Named Entity Recognition",
                "config": """<View>
  <Text name="text" value="$text"/>
  <Labels name="label" toName="text">
    <Label value="Person" background="red"/>
    <Label value="Organization" background="darkorange"/>
    <Label value="Location" background="orange"/>
  </Labels>
</View>""",
            },
            "3": {
                "name": "Image Classification",
                "config": """<View>
  <Image name="image" value="$image"/>
  <Choices name="choice" toName="image">
    <Choice value="Cat"/>
    <Choice value="Dog"/>
    <Choice value="Other"/>
  </Choices>
</View>""",
            },
            "4": {"name": "Custom", "config": ""},
        }

        self.console.print("\n[bold]Available Label Configurations:[/bold]")
        for key, config in label_configs.items():
            self.console.print(f"{key}. {config['name']}")

        label_config_choice = Prompt.ask(
            "Choose label configuration",
            choices=list(label_configs.keys()),
            default="1",
        )

        if label_config_choice == "4":
            label_config = Prompt.ask("Enter custom label configuration XML")
        else:
            label_config = label_configs[label_config_choice]["config"]
            self.console.print(
                f"\n[green]Using {label_configs[label_config_choice]['name']} configuration[/green]"
            )

        for key, method in sampling_configs.items():
            self.console.print(f"{key}. {method}")

        sampling_method_choice = Prompt.ask(
            "Enter sampling method", choices=list(sampling_configs.keys()), default="1"
        )
        sampling_method = sampling_configs[sampling_method_choice]

        # If uncertainty sampling is chosen, setup active learning
        if sampling_method_choice == "3":
            self.console.print(
                f"\n[yellow]Note: Uncertainty sampling requires ML backend setup after project creation[/yellow]"
            )

        self.console.print(
            f"\n[green]Using {sampling_configs[sampling_method_choice]} method[/green]"
        )

        project = self.client.create_project(
            title, description, label_config, sampling_method
        )

        if project:
            self.console.print(f"[green]Project created successfully![/green]")
            self.console.print(f"Project ID: {project.get('id')}")
            self.current_project = project

            # Offer to setup active learning if uncertainty sampling was chosen
            if sampling_method_choice == "3":
                if Confirm.ask(
                    "Would you like to setup active learning for this project?",
                    default=True,
                ):
                    self.setup_active_learning_project()
        else:
            self.console.print("[red]Failed to create project[/red]")

    def create_project_from_config(self, config_path: str):
        """
        Create a new project from a configuration file.

        Args:
            config_path: Path to the JSON configuration file defining the project.
                         Should contain 'title', and optionally 'description',
                         'label_config', and 'sampling'.
        """
        if not self.client:
            self.console.print("[red]No client initialized[/red]")
            return

        try:
            with open(config_path, "r") as f:
                project_config = json.load(f)

            title = project_config.get("title")
            if not title:
                self.console.print(
                    f"[red]Config file {config_path} must contain a 'title' field[/red]"
                )
                return

            description = project_config.get("description", "")
            label_config = project_config.get("label_config", "")

            # If label_config refers to a file, try to load it
            if (
                label_config
                and label_config.endswith(".xml")
                and os.path.exists(label_config)
            ):
                try:
                    with open(label_config, "r") as f:
                        label_config = f.read()
                except Exception as e:
                    self.console.print(
                        f"[yellow]Could not read label config file {label_config}: {e}. Using as string literal.[/yellow]"
                    )

            sampling = project_config.get("sampling", "Sequential sampling")

            self.console.print(
                f"[blue]Creating project '{title}' from config...[/blue]"
            )
            project = self.client.create_project(
                title, description, label_config, sampling
            )

            if project:
                self.console.print(
                    f"[green]Project '{title}' created successfully![/green]"
                )
                self.console.print(f"Project ID: {project.get('id')}")
                self.current_project = project

                # Add ML Backend if present in config
                ml_backend = project_config.get("ml_backend")
                if ml_backend:
                    self.console.print("[blue]Adding ML Backend from config...[/blue]")
                    try:
                        self.client.add_ml_backend(
                            project_id=project.get("id"),
                            url=ml_backend.get("url"),
                            desc=ml_backend.get("description", ""),
                            title=ml_backend.get("title", ""),
                            is_interactive=ml_backend.get("is_interactive", False),
                            auth_method=ml_backend.get("auth_method", "NONE"),
                            basic_auth_user=ml_backend.get("basic_auth_user", "string"),
                            basic_auth_pass=ml_backend.get("basic_auth_pass", "string"),
                            extra_params=ml_backend.get("extra_params", {}),
                        )
                        self.console.print(
                            "[green]ML Backend added successfully![/green]"
                        )
                    except Exception as e:
                        self.console.print(f"[red]Failed to add ML Backend: {e}[/red]")

                # Import initial data if present in config
                initial_data = project_config.get("initial_data")
                if initial_data:
                    self.console.print(
                        "[blue]Importing initial data from config...[/blue]"
                    )

                    data_source_type = initial_data.get("type")
                    data_source = initial_data.get("source")

                    if data_source_type == "url" and data_source:
                        self.client.import_tasks_from_url(
                            project_id=project.get("id"), url=data_source, commit=True
                        )
                    elif data_source_type == "file" and data_source:
                        self.client.import_tasks_from_file(
                            project_id=project.get("id"),
                            file_path=data_source,
                            commit=True,
                        )
                    elif data_source_type == "list" and isinstance(data_source, list):
                        self.client.import_tasks(
                            project_id=project.get("id"), tasks=data_source
                        )
                    else:
                        self.console.print(
                            f"[yellow]Unknown or invalid data source type in config: {data_source_type}[/yellow]"
                        )

                    self.console.print(
                        "[green]Initial data import operation completed.[/green]"
                    )

            else:
                self.console.print("[red]Failed to create project[/red]")

        except FileNotFoundError:
            self.console.print(f"[red]Config file not found: {config_path}[/red]")
        except json.JSONDecodeError:
            self.console.print(f"[red]Invalid JSON in config file: {config_path}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error creating project from config: {e}[/red]")

    def select_project(self):
        """Select a project to work with."""
        if not self.client:
            self.console.print("[red]No client initialized[/red]")
            return

        projects = self.client.get_projects()

        if not projects:
            self.console.print("[yellow]No projects found[/yellow]")
            return

        self.console.print("\n[bold]Available Projects:[/bold]")
        for i, project in enumerate(projects["results"], 1):
            self.console.print(
                f"{i}. {project.get('title', 'Untitled')} (ID: {project.get('id')})"
            )

        try:
            choice = int(Prompt.ask("Select project number", default="1")) - 1
            if 0 <= choice < len(projects["results"]):
                self.current_project = projects["results"][choice]
                self.console.print(
                    f"[green]Selected project: {self.current_project.get('title')}(ID: {self.current_project.get('id')})[/green]"
                )
            else:
                self.console.print("[red]Invalid selection[/red]")
        except ValueError as e:
            print(e)
            self.console.print("[red]Invalid input[/red]")

    def view_project_details(self):
        """View detailed information about the current project."""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        project_id = self.current_project.get("id")
        project = self.client.get_project(project_id)

        if not project:
            self.console.print("[red]Failed to get project details[/red]")
            return

        panel_content = f"""
[bold]Title:[/bold] {project.get('title', 'N/A')}
[bold]Description:[/bold] {project.get('description', 'N/A')}
[bold]ID:[/bold] {project.get('id', 'N/A')}
[bold]Tasks:[/bold] {project.get('task_number', 0)}
[bold]Annotations:[/bold] {project.get('total_annotations_number', 0)}
[bold]Predictions:[/bold] {project.get('total_predictions_number', 0)}
[bold]Created:[/bold] {project.get('created_at', 'N/A')} by {project.get('created_by', 'N/A').get('email')}
[bold]Sampling:[/bold] {project.get('sampling', 'N/A')}
[bold]Model Version (Backend):[/bold] {project.get('model_version', 'Undefined - not added')}
[bold]Start Training on Annotation Update:[/bold] {project.get('start_training_on_annotation_update', 'False')}
        """

        self.console.print(
            Panel(panel_content, title="Project Details", border_style="blue")
        )

        # Show label config if available
        if project.get("label_config"):
            self.console.print("\n[bold]Label Configuration:[/bold]")
            syntax = Syntax(
                project.get("label_config"), "xml", theme="monokai", line_numbers=True
            )
            self.console.print(syntax)

    def list_tasks(self):
        """List tasks for the current project."""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        if self.current_project["task_number"] == 0:
            self.console.print("[yellow]No tasks foundX[/yellow]")
            return

        project_id = self.current_project.get("id")
        tasks = self.client.get_tasks(project_id)

        # get task list from results
        task_cnt = tasks["total"]
        tasks = tasks["tasks"]

        if not tasks:
            self.console.print("[yellow]No tasks foundY[/yellow]")
            return

        table = Table(title=f"Tasks for Project: {self.current_project.get('title')}")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Data Preview", style="green")
        table.add_column("Annotations", style="yellow")
        table.add_column("Created", style="blue")

        for task in tasks[: min(20, len(tasks))]:  # Limit to first 20 tasks
            data_preview = str(task.get("data", {}))
            if len(data_preview) > 50:
                data_preview = data_preview[:50] + "..."

            created_at = task.get("created_at", "")
            if created_at:
                try:
                    created_at = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d %H:%M")
                except:
                    pass

            table.add_row(
                str(task.get("id", "")),
                data_preview,
                str(task.get("total_annotations", 0)),
                created_at,
            )

        self.console.print(table)

        if len(tasks) > 20:
            self.console.print(f"[yellow]Showing first 20 of {task_cnt} tasks[/yellow]")

    def import_tasks_interactive(self):
        """Import tasks to the current project interactively."""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        self.console.print("[bold blue]Import Tasks[/bold blue]")
        import_methods = {
            "1": "Manual entry",
            "2": "JSON file",
            "3": "CSV/TSV file",
            "4": "from URL",
            "5": "Sample data",
        }

        self.console.print("\n[bold]Import methods:[/bold]")
        for key, method in import_methods.items():
            self.console.print(f"{key}. {method}")

        method = Prompt.ask(
            "Choose import method", choices=list(import_methods.keys()), default="1"
        )

        if method == "1":
            self.import_tasks_manual()
        elif method == "2":
            self.import_tasks_from_json_file()
        elif method == "3":
            self.import_tasks_from_csv_file()
        elif method == "4":
            self.import_tasks_from_url()
        elif method == "5":
            self.import_sample_tasks()

    def import_tasks_manual(self):
        """Import tasks through manual entry."""
        tasks = []

        while True:
            task_data = {}

            # Get task fields based on project type
            default_field_value = re.search(
                r'value="\$(\w+)"', self.current_project["label_config"]
            ).group(1)
            print(default_field_value)
            field_name = Prompt.ask(
                "Enter field name (e.g., 'text', 'image', 'audio')",
                default=default_field_value,
            )
            field_value = Prompt.ask(f"Enter {field_name} value")
            task_data[field_name] = field_value

            # Get preannotated fields configuration first
            preannotated_fields = None
            if Confirm.ask("Add preannotated fields for predictions?", default=False):
                fields_input = Prompt.ask(
                    "Enter comma-separated field names for preannotation"
                )
                preannotated_fields = [
                    field.strip() for field in fields_input.split(",") if field.strip()
                ]
                self.console.print(
                    f"[green]Preannotated fields: {preannotated_fields}[/green]"
                )

            # Add preannotated prediction fields
            if preannotated_fields:
                self.console.print(
                    f"\n[bold]Adding prediction data for fields: {preannotated_fields}[/bold]"
                )
                for pred_field in preannotated_fields:
                    pred_value = Prompt.ask(
                        f"Enter prediction value for '{pred_field}'"
                    )
                    task_data[pred_field] = pred_value

            # Add metadata if needed
            if Confirm.ask("Add metadata?"):
                metadata = {}
                while True:
                    key = Prompt.ask("Metadata key (empty to finish)")
                    if not key:
                        break
                    value = Prompt.ask(f"Value for {key}")
                    metadata[key] = value

                if metadata:
                    task_data["meta"] = metadata

            tasks.append(task_data)

            if not Confirm.ask("Add another task?"):
                break

        if tasks:
            project_id = self.current_project.get("id")

            # Advanced options
            commit = Confirm.ask("Commit tasks immediately?", default=True)
            return_task_ids = Confirm.ask("Return task IDs in response?", default=False)

            if self.client.import_tasks(
                project_id, tasks, commit, return_task_ids, preannotated_fields
            ):
                self.console.print(
                    f"[green]Successfully imported {len(tasks)} tasks[/green]"
                )

                # Refresh current project data (interactive mode)
                updated_project = self.client.refresh_project(project_id)
                if updated_project:
                    self.current_project = updated_project
            else:
                self.console.print("[red]Failed to import tasks[/red]")

    def import_tasks_from_csv_file(self):
        """Import tasks from a CSV file."""
        file_path = Prompt.ask("Enter path to CSV file")

        try:
            import csv

            tasks = []

            with open(file_path, "r", encoding="utf-8") as csvfile:
                # Try to detect delimiter
                sample = csvfile.read(1024)
                csvfile.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter

                reader = csv.DictReader(csvfile, delimiter=delimiter)

                for row in reader:
                    # Remove empty values and convert to task format
                    task_data = {k: v for k, v in row.items() if v and v.strip()}
                    if task_data:  # Only add non-empty tasks
                        tasks.append(task_data)

            if not tasks:
                self.console.print("[yellow]No valid tasks found in CSV file[/yellow]")
                return

            # Advanced options
            commit = Confirm.ask("Commit tasks immediately?", default=True)
            return_task_ids = Confirm.ask("Return task IDs in response?", default=False)

            preannotated_fields = None
            if Confirm.ask("Add preannotated fields?", default=False):
                fields_input = Prompt.ask(
                    "Enter comma-separated field names for preannotation"
                )
                preannotated_fields = [
                    field.strip() for field in fields_input.split(",") if field.strip()
                ]

            project_id = self.current_project.get("id")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Importing tasks from CSV...", total=None)

                result = self.client.import_tasks(
                    project_id,
                    tasks,
                    commit=commit,
                    return_task_ids=return_task_ids,
                    preannotated_from_fields=preannotated_fields,
                )

                if result:
                    progress.update(
                        task,
                        description=f"[green]Successfully imported {len(tasks)} tasks[/green]",
                    )
                    progress.stop()

                    # Show import summary
                    summary = f"""[green]Import Summary:[/green]
                    - Tasks processed: {len(tasks)}
                    - File: {file_path}
                    - Commit: {commit}"""

                    self.console.print(summary)

                    if (
                        return_task_ids
                        and isinstance(result, dict)
                        and result.get("task_ids")
                    ):
                        self.console.print(
                            f"Task IDs: {result['task_ids'][:10]}{'...' if len(result['task_ids']) > 10 else ''}"
                        )
                else:
                    progress.update(
                        task, description="[red]Failed to import tasks[/red]"
                    )
                    progress.stop()

        except FileNotFoundError:
            self.console.print(f"[red]File not found: {file_path}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error reading CSV file: {e}[/red]")

    def import_tasks_from_json_file(self):
        """Import tasks from a JSON file."""
        file_path = Prompt.ask("Enter path to JSON file")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tasks = json.load(f)

            if not isinstance(tasks, list):
                tasks = [tasks]

            # Advanced options
            commit = Confirm.ask("Commit tasks immediately?", default=True)
            return_task_ids = Confirm.ask("Return task IDs in response?", default=False)

            preannotated_fields = None
            if Confirm.ask("Add preannotated fields?", default=False):
                fields_input = Prompt.ask(
                    "Enter comma-separated field names for preannotation"
                )
                preannotated_fields = [
                    field.strip() for field in fields_input.split(",") if field.strip()
                ]

            project_id = self.current_project.get("id")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Importing tasks from JSON...", total=None)

                result = self.client.import_tasks(
                    project_id,
                    tasks,
                    commit=commit,
                    return_task_ids=return_task_ids,
                    preannotated_from_fields=preannotated_fields,
                )

                if result:
                    progress.update(
                        task,
                        description=f"[green]Successfully imported {len(tasks)} tasks[/green]",
                    )
                    progress.stop()
                    self.console.print(
                        f"[green]Successfully imported {len(tasks)} tasks from {file_path}[/green]"
                    )
                else:
                    progress.update(
                        task, description="[red]Failed to import tasks[/red]"
                    )
                    progress.stop()

        except FileNotFoundError:
            self.console.print(f"[red]File not found: {file_path}[/red]")
        except json.JSONDecodeError as e:
            self.console.print(f"[red]Invalid JSON format: {e}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error reading JSON file: {e}[/red]")

    def import_sample_tasks(self):
        """Import sample tasks for demonstration."""
        sample_tasks = [
            {"text": "This is a positive example of sentiment analysis."},
            {"text": "This is a negative example that shows bad sentiment."},
            {"text": "This is a neutral statement without strong sentiment."},
            {"text": "I love this product! It's amazing and works perfectly."},
            {"text": "This product is terrible and doesn't work at all."},
        ]

        project_id = self.current_project.get("id")
        if self.client.import_tasks(project_id, sample_tasks):
            self.console.print(
                f"[green]Successfully imported {len(sample_tasks)} sample tasks[/green]"
            )
        else:
            self.console.print("[red]Failed to import sample tasks[/red]")

    def import_tasks_from_json_file(self):
        """Import tasks from a JSON file."""
        file_path = Prompt.ask("Enter path to JSON file")

        # Advanced options
        commit = Confirm.ask("Commit tasks immediately?", default=True)
        return_task_ids = Confirm.ask("Return task IDs in response?", default=False)

        preannotated_fields = None
        if Confirm.ask("Add preannotated fields?", default=False):
            fields_input = Prompt.ask(
                "Enter comma-separated field names for preannotation"
            )
            preannotated_fields = [
                field.strip() for field in fields_input.split(",") if field.strip()
            ]

        project_id = self.current_project.get("id")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Importing tasks from file...", total=None)

            result = self.client.import_tasks_from_file(
                project_id,
                file_path,
                commit=commit,
                return_task_ids=return_task_ids,
                preannotated_from_fields=preannotated_fields,
            )

            if result:
                progress.update(
                    task, description=f"[green]Successfully imported tasks[/green]"
                )
                progress.stop()

                # Show import summary
                summary = f"""[green]Import Summary:[/green]
                - Tasks added: {result.get('task_count', 'N/A')}
                - Annotations added: {result.get('annotation_count', 'N/A')}
                - Predictions added: {result.get('prediction_count', 'N/A')}
                - Time taken: {result.get('duration', 'N/A')} seconds"""

                self.console.print(summary)

                if return_task_ids and result.get("task_ids"):
                    self.console.print(
                        f"Task IDs: {result['task_ids'][:10]}{'...' if len(result['task_ids']) > 10 else ''}"
                    )
            else:
                progress.update(task, description="[red]Failed to import tasks[/red]")
                progress.stop()

    def import_tasks_from_url(self):
        """Import tasks from a URL."""
        url = Prompt.ask("Enter URL to tasks file")

        # Advanced options
        commit = Confirm.ask("Commit tasks immediately?", default=True)
        return_task_ids = Confirm.ask("Return task IDs in response?", default=False)

        project_id = self.current_project.get("id")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Importing tasks from URL...", total=None)

            result = self.client.import_tasks_from_url(
                project_id, url, commit=commit, return_task_ids=return_task_ids
            )

            if result:
                progress.update(
                    task,
                    description=f"[green]Successfully imported tasks from URL[/green]",
                )
                progress.stop()
                self.console.print(
                    f"[green]Successfully imported {result.get('task_count', 'N/A')} tasks from {url}[/green]"
                )
            else:
                progress.update(
                    task, description="[red]Failed to import tasks from URL[/red]"
                )
                progress.stop()

    def export_annotations(self):
        """Export annotations from the current project."""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        export_formats = ["JSON", "CSV", "TSV", "YOLO", "COCO"]

        self.console.print("\n[bold]Available export formats:[/bold]")
        for i, fmt in enumerate(export_formats, 1):
            self.console.print(f"{i}. {fmt}")

        try:
            choice = int(Prompt.ask("Choose export format", default="1")) - 1
            if 0 <= choice < len(export_formats):
                export_format = export_formats[choice]
            else:
                export_format = "JSON"
        except ValueError:
            export_format = "JSON"

        project_id = self.current_project.get("id")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Exporting annotations...", total=None)

            export_data = self.client.export_annotations(project_id, export_format)

            if export_data:
                filename = f"project_{project_id}_annotations.{export_format.lower()}"

                try:
                    with open(filename, "w") as f:
                        f.write(export_data)

                    progress.update(
                        task, description=f"[green]Export completed![/green]"
                    )
                    progress.stop()
                    self.console.print(
                        f"[green]Annotations exported to {filename}[/green]"
                    )

                except Exception as e:
                    progress.update(task, description=f"[red]Export failed![/red]")
                    progress.stop()
                    self.console.print(f"[red]Failed to save export: {e}[/red]")
            else:
                progress.update(task, description=f"[red]Export failed![/red]")
                progress.stop()
                self.console.print("[red]Failed to export annotations[/red]")

    def delete_project(self):
        """Delete the current project or all projects based on user selection."""

        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            if Confirm.ask("Do you want to delete all projects?"):
                self.delete_all_projects()
            return

        # Show deletion options
        project_title = self.current_project.get("title", "Unknown")
        project_id = self.current_project.get("id")

        delete_options = {
            "1": f"Delete selected project '{project_title}' (ID: {project_id})",
            "2": "Delete all projects",
        }

        for key, option in delete_options.items():
            self.console.print(f"{key}. {option}")

        choice = Prompt.ask("Enter delete option", choices=["1", "2"], default="1")

        if choice == "1":
            if Confirm.ask(
                f"Are you sure you want to delete project '{project_title}'?"
            ):

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    task = progress.add_task(
                        "[bold green]Deleting all Projects...", total=None
                    )

                    if self.client.delete_project(project_id):
                        progress.update(
                            task,
                            description=f"[green]Project '{project_title}' deleted successfully[/green]",
                        )
                        self.current_project = None
                    else:
                        progress.update(
                            task, description="[red]Failed to delete project[/red]"
                        )
        else:
            self.delete_all_projects()

    def delete_all_projects(self):
        """Delete all projects with individual confirmations."""
        projects = self.client.get_projects()

        if not projects or not projects.get("results"):
            self.console.print("[yellow]No projects found[/yellow]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("[bold green]Deleting all Projects...", total=None)

            for project in projects["results"]:
                project_title = project.get("title", "Unknown")
                if Confirm.ask(f"Delete project '{project_title}'?"):
                    if self.client.delete_project(project.get("id")):
                        progress.update(
                            task,
                            description=f"[green]Project '{project_title}' deleted successfully[/green]",
                        )
                        # Clear current project if it was the one deleted
                        if self.current_project and self.current_project.get(
                            "id"
                        ) == project.get("id"):
                            self.current_project = None
                    else:
                        progress.update(
                            task,
                            description=f"[red]Failed to delete project '{project_title}'[/red]",
                        )

    def list_ml_backend(self):
        """List ML Backend for selected project"""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("[bold green]Fetching ML backends...", total=None)

            existing_backends = self.client.list_ml_backends(
                self.current_project.get("id")
            )

            if existing_backends is not None:
                if existing_backends:
                    progress.update(
                        task,
                        description=f"[green]{len(existing_backends)} ML Backends found[/green]",
                    )

                    table = Table(
                        title="Existing ML Backends added to selected project"
                    )
                    table.add_column("ID", style="cyan", no_wrap=True)
                    table.add_column("Title", style="magenta")
                    table.add_column("Description", style="green")
                    table.add_column("URL", style="green")
                    table.add_column("Model Version", style="yellow")
                    table.add_column("Is interactive", style="yellow")
                    table.add_column("created at", style="yellow")
                    table.add_column("updated at", style="yellow")
                    table.add_column("Status", style="red")

                    for backend in existing_backends:
                        try:
                            created_at = datetime.fromisoformat(
                                backend.get("created_at", "N/A").replace("Z", "+00:00")
                            ).strftime("%Y-%m-%d %H:%M")
                            updated_at = datetime.fromisoformat(
                                backend.get("updated_at", "N/A").replace("Z", "+00:00")
                            ).strftime("%Y-%m-%d %H:%M")
                        except:
                            pass
                        table.add_row(
                            str(backend.get("id", "N/A")),
                            backend.get("title", "No Title"),
                            backend.get("description", "N/A"),
                            backend.get("url", "N/A"),
                            backend.get("model_version", "N/A"),
                            str(backend.get("is_interactive", "N/A")),
                            created_at,
                            updated_at,
                            backend.get("readable_state", "Unknown"),
                        )
                    self.console.print(table)
                else:
                    progress.update(
                        task,
                        description="[yellow] No existing ML backends found[/yellow]",
                    )
                    progress.stop()
            else:
                progress.update(
                    task, description="[red] Failed to list ML backends[/red]"
                )
                progress.stop()
                return

    def delete_ml_backend(self):
        """Get ML Backend Details for selected project"""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        existing_backends = self.client.list_ml_backends(self.current_project.get("id"))

        if existing_backends is not None:
            self.list_ml_backend()
            # Extract the IDs and add "all" option
            backend_ids = ["all"] + [
                str(backend["id"]) for backend in existing_backends
            ]
            ml_backend_choice = Prompt.ask(
                "Which backend(s) would you like to delete?", choices=backend_ids
            )

            if ml_backend_choice == "all":
                # Use all backend IDs (excluding "all" option)
                ids_to_delete = [str(backend["id"]) for backend in existing_backends]
            else:
                ids_to_delete = [ml_backend_choice]

            # Process each ID to delete
            for backend_id in ids_to_delete:
                self.console.print(
                    f"\n[bold cyan]Removing ML backend {backend_id}...[/bold cyan]"
                )

                if Confirm.ask(
                    f"Are you sure you want to remove ML backend id {backend_id} from project '{self.current_project.get('id')}'?"
                ):
                    removal_success = self.client.remove_ml_backend(
                        self.current_project["id"], backend_id
                    )

                    if removal_success:
                        self.console.print(
                            f"[green]Successfully removed ML backend {backend_id}[/green]"
                        )
                    else:
                        self.console.print(
                            f"[red]Failed to remove ML backend {backend_id}[/red]"
                        )
                else:
                    self.console.print(
                        f"[yellow]Skipped removing ML backend {backend_id}[/yellow]"
                    )

        else:
            self.console.print("[yellow]No existing ML backends found[/yellow]")
            return

    def add_ml_backend(self):
        """Add ML Backend for selected project"""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return
        # TODO not self.client before not self.current_project? or delete it in all functions
        if not self.client:
            self.console.print("[red]No client initialized[/red]")
            return

        backend_url_example = {
            "0": "Manual URL input...",
            "1": "http://sklearn_text_classifier:9090",
            "2": "http://yolo:9090",
        }

        self.console.print("[bold blue]Adding ML Backend[/bold blue]")
        self.console.print(
            f"Project: [cyan]{self.current_project.get('title', 'Unknown')}[/cyan] (ID: {self.current_project.get('id')})"
        )

        # Set basic ML backend information
        for key, url_example in backend_url_example.items():
            self.console.print(f"{key}. {url_example}")

        url = Prompt.ask(
            "Choose backend url", choices=list(backend_url_example.keys()), default="0"
        )
        if url != "0":
            url = backend_url_example[url]
        else:
            url = Prompt.ask("Enter ML backend URL")

        title = Prompt.ask("Enter ML backend title", default="")
        description = Prompt.ask("Enter ML backend description", default="")

        # Interactive mode
        is_interactive = Confirm.ask("Enable interactive mode?", default=False)

        # Authentication methods
        auth_methods = {"1": "NONE", "2": "BASIC_AUTH", "3": "API_KEY"}

        self.console.print("\n[bold]Authentication Methods:[/bold]")

        auth_display = {
            "NONE": "No Authentication",
            "BASIC_AUTH": "HTTP Basic Authentication",
            "API_KEY": "API Key Authentication",
        }

        for key, method in auth_methods.items():
            self.console.print(f"{key}. {auth_display[method]}")

        auth_choice = Prompt.ask(
            "Choose authentication method",
            choices=list(auth_methods.keys()),
            default="1",
        )
        auth_method = auth_methods[auth_choice]

        # Authentication credentials
        basic_auth_user = "string"
        basic_auth_pass = "string"

        if auth_method == "BASIC_AUTH":
            self.console.print("\n[yellow]Basic Authentication Setup:[/yellow]")
            basic_auth_user = Prompt.ask("Enter username")
            basic_auth_pass = Prompt.ask("Enter password", password=True)

        # Extra parameters
        extra_params = {}
        if Confirm.ask("Add extra parameters?", default=False):
            self.console.print(
                "\n[dim]Enter extra parameters as key=value pairs. Press Enter with empty key to finish.[/dim]"
            )
            while True:
                key = Prompt.ask("Parameter key", default="")
                if not key:
                    break
                value = Prompt.ask(f"Value for '{key}'")
                # Try to convert to appropriate type
                try:
                    # Try int first
                    if value.isdigit():
                        value = int(value)
                    # Try bool
                    elif value.lower() in ["true", "false"]:
                        value = value.lower() == "true"
                    # Try float
                    elif "." in value and value.replace(".", "").isdigit():
                        value = float(value)
                except:
                    pass  # Keep as string

                extra_params[key] = value

        # Summary
        self.console.print("\n[bold]ML Backend Configuration Summary:[/bold]")
        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Property", style="cyan")
        summary_table.add_column("Value", style="white")
        summary_table.add_row("URL", url)
        summary_table.add_row("Title", title or "[dim]Auto-generated[/dim]")
        summary_table.add_row("Description", description or "[dim]None[/dim]")
        summary_table.add_row("Interactive", str(is_interactive))
        summary_table.add_row("Authentication", auth_method)
        if auth_method == "BASIC_AUTH":
            summary_table.add_row("Username", basic_auth_user)
            summary_table.add_row("Password", "[dim]Hidden[/dim]")
        if extra_params:
            summary_table.add_row("Extra Params", str(extra_params))

        self.console.print(summary_table)

        # Confirmation
        if not Confirm.ask("\nProceed with adding this ML backend?", default=True):
            self.console.print("[yellow]Cancelled[/yellow]")
            return

        # Add the ML backend
        with self.console.status("[bold green]Adding ML backend..."):
            backend = self.client.add_ml_backend(
                project_id=self.current_project.get("id"),
                url=url,
                desc=description,
                title=title if title else None,
                is_interactive=is_interactive,
                auth_method=auth_method,
                basic_auth_user=basic_auth_user,
                basic_auth_pass=basic_auth_pass,
                extra_params=extra_params,
            )

        if backend:
            # Success panel
            success_text = f"""[green] ML Backend added successfully![/green]
                            [bold]Details:[/bold]
                            - Backend ID: [cyan]{backend.get('id')}[/cyan]
                            - Title: [magenta]{backend.get('title', 'N/A')}[/magenta]
                            - URL: [green]{backend.get('url')}[/green]
                            - Status: [yellow]{backend.get('readable_state', 'Unknown')}[/yellow]
                            - Interactive: {backend.get('is_interactive')}"""

            self.console.print(Panel(success_text, border_style="green"))

            # Refresh current project data (interactive mode)
            updated_project = self.client.refresh_project(
                self.current_project.get("id")
            )
            if updated_project:
                self.current_project = updated_project

        else:
            self.console.print("[red]Failed to add ML backend")

    def list_import_storage(self):
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        storage = self.client.get_all_import_storage(self.current_project["id"])

        if not storage:
            self.console.print("[yellow]No storages found[/yellow]")
            return

        table = Table(
            title=f"Import Storage Configurations for project wiht id {self.current_project['id']}"
        )
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Type", style="red")
        table.add_column("Title", style="magenta")
        table.add_column("Description", style="green")
        table.add_column("Regex Filter", style="bright_magenta")
        table.add_column("Use Blob URLs", style="bright_cyan")
        table.add_column("Sync Count", style="blue")
        table.add_column("Last Synced", style="bright_green")
        table.add_column("Created", style="bright_black")
        table.add_column("Status", style="yellow")

        for storage_config in storage:
            created_at = storage_config.get("created_at", "")
            if created_at:
                try:
                    created_at = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d %H:%M")
                except:
                    pass

            last_sync = storage_config.get("last_sync", "")
            if last_sync:
                try:
                    last_sync = datetime.fromisoformat(
                        last_sync.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d %H:%M")
                except:
                    pass

            description = storage_config.get("description", "")
            truncated_desc = (
                description[:30] + "..." if len(description) > 30 else description
            )

            regex_filter = storage_config.get("regex_filter", "")
            truncated_regex = (
                regex_filter[:15] + "..." if len(regex_filter) > 15 else regex_filter
            )

            table.add_row(
                str(storage_config.get("id", "")),
                storage_config.get("type", ""),
                storage_config.get("title", ""),
                truncated_desc,
                truncated_regex,
                str(storage_config.get("use_blob_urls", False)),
                str(storage_config.get("last_sync_count", 0)),
                last_sync or "Never",
                created_at,
                storage_config.get("status", ""),
            )

        self.console.print(table)

    def create_import_storage(self):
        """Add Local Storage for selected project"""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return
        # TODO not self.client before not self.current_project? or delete it in all functions
        if not self.client:
            self.console.print("[red]No client initialized[/red]")
            return

        self.console.print("[bold blue]Adding Local Storage[/bold blue]")
        self.console.print(
            f"Project: [cyan]{self.current_project.get('title', 'Unknown')}[/cyan] (ID: {self.current_project.get('id')})"
        )

        title = Prompt.ask("Enter local storage title", default="TEST")
        description = Prompt.ask("Enter local storage description", default="")

        path = Prompt.ask("Add local storage path eg. /label-studio/data/...")
        # TODO remove hardcoded path
        # path = "/label-studio/data/datasets_cat_dog"
        self.console.print("\n[bold]Regex Filters:[/bold]")

        regex_filter_choice = {
            "0": "None",
            "1": ".*csv",
            "2": ".*(jpe?g|png|tiff)",
            "3": ".*json",
        }

        for key, method in regex_filter_choice.items():
            self.console.print(f"{key}. {method}")

        regex_filter = Prompt.ask(
            "Choose regex filter: ",
            choices=list(regex_filter_choice.keys()),
            default="0",
        )
        regex_filter = regex_filter_choice[regex_filter]
        # TODO blob_urls settings
        use_blob_urls = True

        # Summary
        self.console.print("\n[bold]Local Storage Configuration Summary:[/bold]")
        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Property", style="cyan")
        summary_table.add_column("Value", style="white")
        summary_table.add_row("Title", title or "[dim]Auto-generated[/dim]")
        summary_table.add_row("Description", description or "[dim]None[/dim]")
        summary_table.add_row("path", path)
        summary_table.add_row("regex_filter", regex_filter)
        summary_table.add_row("use_blob_urls", str(use_blob_urls))

        self.console.print(summary_table)

        # Confirmation
        if not Confirm.ask("\nProceed with adding this local storage?", default=True):
            self.console.print("[yellow]Cancelled[/yellow]")
            return

        # TODO validate before creating
        # Add local storage
        with self.console.status("[bold green]Creating local storage..."):
            storage = self.client.create_import_storage(
                project_id=self.current_project.get("id"),
                desc=description,
                title=title,
                path=path,
                regex_filter=regex_filter,
                use_blob_urls=use_blob_urls,
            )
        if storage:
            print(storage)
            # Success panel
            success_text = f"""[green]Storage added successfully![/green]
[bold]Details:[/bold]
    - Storage ID: [cyan]{storage.get('id')}[/cyan]
    - Title: [magenta]{storage.get('title', 'N/A')}[/magenta]
    - Type: [green]{storage.get('type')}[/green]
    - Path: [yellow]{storage.get('path', 'Unknown')}[/yellow]
    - Regex Filter: [yellow]{storage.get('regex_filter', 'Unknown')}[/yellow]
    - Last Sync: [magenta]{storage.get('last_sync', 'N/A')}[/magenta]
    - Status: [red]{storage.get('status')}[/red]"""

            self.console.print(Panel(success_text, border_style="green"))

            # Sync local storage
            if Confirm.ask(
                "\nProceed with syncing files from local storage?", default=True
            ):

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    task = progress.add_task("Syncing local storage...", total=None)

                    synced_storage = self.client.sync_import_storage(storage["id"])
                    if synced_storage:
                        success_text = f"""[green]Storage synced successfully![/green]
[bold]Details:[/bold]
    - Storage ID: [cyan]{synced_storage.get('id')}[/cyan]
    - Last Sync: [magenta]{synced_storage.get('last_sync', 'N/A')}[/magenta]
    - Synced Tasks: [magenta]{synced_storage.get('last_sync_count', 'None')}[/magenta]
    - Status: [green]{synced_storage.get('status')}[/green]"""

                        self.console.print(Panel(success_text, border_style="green"))
                    else:
                        progress.update(
                            task, description="[red]Failed to sync local storage[/red]"
                        )
                        progress.stop()

        else:
            self.console.print("[red]Failed to add local storage[/red]")

    def delete_import_storage(self):
        """Add Local Storage for selected project"""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return
        # TODO not self.client before not self.current_project? or delete it in all functions
        if not self.client:
            self.console.print("[red]No client initialized[/red]")
            return

        existing_storages = self.client.get_all_import_storage(
            self.current_project["id"]
        )

        if existing_storages:
            self.list_import_storage()
            # Extract the IDs and add "all" option
            storage_ids = ["all"] + [
                str(storage["id"]) for storage in existing_storages
            ]
            storage_choice = Prompt.ask(
                "Which storage(s) would you like to delete?", choices=storage_ids
            )

            if storage_choice == "all":
                # Use all storage IDs (excluding "all" option)
                ids_to_delete = [str(storage["id"]) for storage in existing_storages]
            else:
                ids_to_delete = [storage_choice]

            # Process each ID to delete
            for storage_id in ids_to_delete:
                self.console.print(
                    f"\n[bold cyan]Removing local storage {storage_id}...[/bold cyan]"
                )

                if Confirm.ask(
                    f"Are you sure you want to remove local storage id {storage_id} from project '{self.current_project.get('id')}'?"
                ):
                    removal_success = self.client.remove_import_storage(
                        self.current_project["id"], storage_id
                    )

                    if removal_success:
                        self.console.print(
                            f"[green]Successfully removed local storage {storage_id}[/green]"
                        )
                    else:
                        self.console.print(
                            f"[red]Failed to remove local storage {storage_id}[/red]"
                        )
                else:
                    self.console.print(
                        f"[yellow]Skipped removing local storage {storage_id}[/yellow]"
                    )

        else:
            self.console.print("[yellow]No existing local storages found[/yellow]")
            return

    def delete_task(self):
        """Delete all or single tasks from current project based on user selection."""

        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        if self.current_project["task_number"] == 0:
            self.console.print("[yellow]No tasks found[/yellow]")
            return

        # Show deletion options
        project_title = self.current_project.get("title", "Unknown")
        project_id = self.current_project.get("id")

        delete_options = {
            "1": f"Delete all tasks from current project '{project_title}' (ID: {project_id})",
            "2": "Delete a specific task from current project",
        }

        for key, option in delete_options.items():
            self.console.print(f"{key}. {option}")

        choice = Prompt.ask("Enter delete option", choices=["1", "2"], default="2")

        if choice == "1":
            if Confirm.ask(
                f"Are you sure you want to delete the first 100 tasks from current project '{project_title}'?"
            ):

                tasks = self.client.get_tasks(project_id)

                # get task list from results
                tasks = tasks["tasks"]

                if not tasks:
                    self.console.print("[yellow]No tasks found[/yellow]")
                    return

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    p_task = progress.add_task("Deleting all tasks...", total=None)
                    if self.client.delete_all_tasks(project_id):
                        progress.update(
                            p_task,
                            description=f"[green]All tasks deleted successfully[/green] ",
                        )
                    else:
                        progress.update(
                            p_task,
                            description=f"[red]Failed to delete all tasks'[/red]",
                        )
                        progress.stop()

        else:
            task_id = Prompt.ask("Enter the ID of the task to be deleted: ")
            if self.client.delete_tasks(task_id):
                self.console.print(
                    f"[green]Task '{task_id}' deleted successfully[/green]"
                )
            else:
                self.console.print(
                    f"[red]Failed to delete task with {task_id} - maybe there is no task with this id!'[/red]"
                )

    def train_ml_backend(self):
        """Train ML backend for selected project."""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        # Get existing ML backends
        existing_backends = self.client.list_ml_backends(self.current_project.get("id"))

        if not existing_backends:
            self.console.print("[yellow]No ML backends found for this project[/yellow]")
            return

        # Display available backends
        self.console.print("\n[bold]Available ML Backends:[/bold]")
        backend_choices = {}
        for i, backend in enumerate(existing_backends, 1):
            backend_id = str(backend["id"])
            backend_choices[str(i)] = backend_id
            status = backend.get("readable_state", "Unknown")
            self.console.print(
                f"{i}. {backend.get('title', 'Untitled')} (ID: {backend_id}) - Status: {status}"
            )

        # Let user select backend to train
        choice = Prompt.ask(
            "Select ML backend to train", choices=list(backend_choices.keys())
        )
        selected_backend_id = int(backend_choices[choice])

        # Confirm training
        backend_title = next(
            b.get("title", "Untitled")
            for b in existing_backends
            if b["id"] == selected_backend_id
        )

        if not Confirm.ask(
            f"Start training for '{backend_title}' (ID: {selected_backend_id})?",
            default=True,
        ):
            self.console.print("[yellow]Training cancelled[/yellow]")
            return

        # Start training
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Starting ML backend training...", total=None)

            result = self.client.train_ml_backend(selected_backend_id)

            if result:
                progress.update(
                    task, description=f"[green]Training started successfully[/green]"
                )
                progress.stop()

                # Show training details
                training_info = f"""[green]Training Started Successfully![/green]
                
    [bold]Details:[/bold]
    - Backend ID: [cyan]{selected_backend_id}[/cyan]
    - Backend Title: [magenta]{backend_title}[/magenta]
    - Training Status: [yellow]{result.get('status', 'Started')}[/yellow]
    - Message: [green]{result.get('message', 'Training initiated')}[/green]"""

                self.console.print(Panel(training_info, border_style="green"))

                # Offer to check training status
                if Confirm.ask(
                    "Would you like to check training status?", default=True
                ):
                    self.check_training_status(selected_backend_id)

            else:
                progress.update(task, description="[red]Failed to start training[/red]")
                progress.stop()

    def check_training_status(self, backend_id: int = None):
        """Check training status for ML backend."""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        if backend_id is None:
            # Get existing ML backends if no specific ID provided
            existing_backends = self.client.list_ml_backends(
                self.current_project.get("id")
            )

            if not existing_backends:
                self.console.print(
                    "[yellow]No ML backends found for this project[/yellow]"
                )
                return

            # Display available backends
            self.console.print("\n[bold]Available ML Backends:[/bold]")
            backend_choices = {}
            for i, backend in enumerate(existing_backends, 1):
                backend_id_str = str(backend["id"])
                backend_choices[str(i)] = backend_id_str
                status = backend.get("readable_state", "Unknown")
                self.console.print(
                    f"{i}. {backend.get('title', 'Untitled')} (ID: {backend_id_str}) - Status: {status}"
                )

            choice = Prompt.ask(
                "Select ML backend to check status",
                choices=list(backend_choices.keys()),
            )
            backend_id = int(backend_choices[choice])

        # Get training status
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Checking training status...", total=None)

            status_info = self.client.get_training_status(backend_id)

            if status_info:
                progress.update(task, description=f"[green]Status retrieved[/green]")
                progress.stop()

                # Display status information with safe key access
                status_panel = f"""[bold]ML Backend Training Status[/bold]
                
    - Backend ID: [cyan]{backend_id}[/cyan]
    - Title: [magenta]{status_info.get('title', 'N/A')}[/magenta]
    - Status: [yellow]{status_info.get('readable_state', status_info.get('status', 'Unknown'))}[/yellow]
    - Model Version: [green]{status_info.get('model_version', 'N/A')}[/green]
    - Last Training: [blue]{status_info.get('updated_at', 'N/A')}[/blue]
    - Interactive Mode: [cyan]{status_info.get('is_interactive', False)}[/cyan]
    - URL: [green]{status_info.get('url', 'N/A')}[/green]"""

                self.console.print(Panel(status_panel, border_style="blue"))

            else:
                progress.update(
                    task, description="[red]Failed to get training status[/red]"
                )
                progress.stop()

    def list_predictions(self):
        """List predictions for the current project or specific task."""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        # Ask user for filtering options
        filter_options = {
            "1": "All predictions for current project",
            "2": "Predictions for specific task",
            "3": "All predictions (no filter)",
        }

        self.console.print("\n[bold]Filter Options:[/bold]")
        for key, option in filter_options.items():
            self.console.print(f"{key}. {option}")

        choice = Prompt.ask(
            "Choose filter option", choices=list(filter_options.keys()), default="1"
        )

        task_id = None
        project_id = None

        if choice == "1":
            project_id = self.current_project.get("id")
        elif choice == "2":
            task_id = int(Prompt.ask("Enter task ID"))
        # choice == "3" means no filter (both None)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Fetching predictions...", total=None)

            predictions = self.client.list_predictions(
                task_id=task_id, project_id=project_id
            )

            if predictions:
                progress.update(
                    task,
                    description=f"[green]Found {len(predictions)} predictions[/green]",
                )
                progress.stop()

                if predictions:
                    table = Table(title="Predictions")
                    table.add_column("ID", style="cyan", no_wrap=True)
                    table.add_column("Task ID", style="magenta")
                    table.add_column("Model Version", style="green")
                    table.add_column("Score", style="yellow")
                    table.add_column("Created", style="blue")

                    for pred in predictions[:20]:  # Limit to first 20
                        created_at = pred.get("created_at", "")
                        if created_at:
                            try:
                                created_at = datetime.fromisoformat(
                                    created_at.replace("Z", "+00:00")
                                ).strftime("%Y-%m-%d %H:%M")
                            except:
                                pass

                        table.add_row(
                            str(pred.get("id", "")),
                            str(pred.get("task", "")),
                            pred.get("model_version", "N/A"),
                            str(pred.get("score", "N/A")),
                            created_at,
                        )

                    self.console.print(table)

                    if len(predictions) > 20:
                        self.console.print(
                            f"[yellow]Showing first 20 of {len(predictions)} predictions[/yellow]"
                        )
                else:
                    self.console.print("[yellow]No predictions found[/yellow]")
            else:
                progress.update(
                    task, description="[red]Failed to fetch predictions[/red]"
                )
                progress.stop()

    def create_prediction_interactive(self):
        """Create a prediction interactively."""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        self.console.print("[bold blue]Create Prediction[/bold blue]")

        # Get task ID
        task_id = int(Prompt.ask("Enter task ID"))

        # Get prediction result (simplified JSON input)
        self.console.print("\n[yellow]Enter prediction result in JSON format:[/yellow]")
        self.console.print(
            '[dim]Example: [{"value": {"choices": ["positive"]}, "from_name": "sentiment", "to_name": "text", "type": "choices"}][/dim]'
        )

        result_json = Prompt.ask("Prediction result JSON")

        try:
            result = json.loads(result_json)
        except json.JSONDecodeError as e:
            self.console.print(f"[red]Invalid JSON format: {e}[/red]")
            return

        # Optional parameters
        score = None
        if Confirm.ask("Add prediction score?", default=False):
            score = float(Prompt.ask("Enter score (0.0-1.0)", default="0.5"))

        model_version = None
        if Confirm.ask("Add model version?", default=False):
            model_version = Prompt.ask("Enter model version")

        # Create prediction
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Creating prediction...", total=None)

            prediction = self.client.create_prediction(
                task_id, result, score, model_version
            )

            if prediction:
                progress.update(
                    task, description=f"[green]Prediction created successfully[/green]"
                )
                progress.stop()

                success_info = f"""[green]Prediction Created Successfully![/green]
                
    [bold]Details:[/bold]
    - Prediction ID: [cyan]{prediction.get('id')}[/cyan]
    - Task ID: [magenta]{prediction.get('task')}[/magenta]
    - Model Version: [yellow]{prediction.get('model_version', 'N/A')}[/yellow]
    - Score: [green]{prediction.get('score', 'N/A')}[/green]"""

                self.console.print(Panel(success_info, border_style="green"))
            else:
                progress.update(
                    task, description="[red]Failed to create prediction[/red]"
                )
                progress.stop()

    def delete_prediction_interactive(self):
        """Delete predictions interactively."""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        # First show available predictions
        predictions = self.client.list_predictions(
            project_id=self.current_project.get("id")
        )

        if not predictions:
            self.console.print("[yellow]No predictions found for this project[/yellow]")
            return

        self.console.print("\n[bold]Available Predictions:[/bold]")
        prediction_choices = {}
        for i, pred in enumerate(predictions[:20], 1):  # Limit to first 20
            pred_id = str(pred["id"])
            prediction_choices[str(i)] = pred_id
            task_id = pred.get("task", "N/A")
            score = pred.get("score", "N/A")
            model_version = pred.get("model_version", "N/A")
            self.console.print(
                f"{i}. ID: {pred_id} | Task: {task_id} | Score: {score} | Model: {model_version}"
            )

        # Add "all" option
        prediction_choices["all"] = "all"
        self.console.print("all. Delete all predictions")

        choice = Prompt.ask(
            "Select prediction(s) to delete", choices=list(prediction_choices.keys())
        )

        if choice == "all":
            if not Confirm.ask(
                f"Are you sure you want to delete ALL {len(predictions)} predictions?",
                default=False,
            ):
                self.console.print("[yellow]Cancelled[/yellow]")
                return
            ids_to_delete = [pred["id"] for pred in predictions]
        else:
            ids_to_delete = [int(prediction_choices[choice])]

        # Delete predictions
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Deleting predictions...", total=None)

            success_count = 0
            for pred_id in ids_to_delete:
                if self.client.delete_prediction(pred_id):
                    success_count += 1

            if success_count == len(ids_to_delete):
                progress.update(
                    task,
                    description=f"[green]Successfully deleted {success_count} predictions[/green]",
                )
            else:
                progress.update(
                    task,
                    description=f"[yellow]Deleted {success_count}/{len(ids_to_delete)} predictions[/yellow]",
                )

            progress.stop()

    def view_prediction_details(self):
        """View detailed information about a specific prediction."""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        prediction_id = int(Prompt.ask("Enter prediction ID"))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Fetching prediction details...", total=None)

            prediction = self.client.get_prediction_details(prediction_id)

            if prediction:
                progress.update(
                    task, description=f"[green]Prediction details retrieved[/green]"
                )
                progress.stop()

                # Display prediction details
                details_panel = f"""[bold]Prediction Details[/bold]
                
    - Prediction ID: [cyan]{prediction.get('id')}[/cyan]
    - Task ID: [magenta]{prediction.get('task')}[/magenta]
    - Model Version: [yellow]{prediction.get('model_version', 'N/A')}[/yellow]
    - Score: [green]{prediction.get('score', 'N/A')}[/green]
    - Created: [blue]{prediction.get('created_at', 'N/A')}[/blue]
    - Updated: [blue]{prediction.get('updated_at', 'N/A')}[/blue]"""

                self.console.print(Panel(details_panel, border_style="blue"))

                # Show prediction result if available
                if prediction.get("result"):
                    self.console.print("\n[bold]Prediction Result:[/bold]")
                    syntax = Syntax(
                        json.dumps(prediction["result"], indent=2),
                        "json",
                        theme="monokai",
                        line_numbers=True,
                    )
                    self.console.print(syntax)
            else:
                progress.update(
                    task, description="[red]Failed to get prediction details[/red]"
                )
                progress.stop()

    def open_browser_interface(self, task_id):
        """Open Label Studio interface in browser."""
        import webbrowser

        project_id = self.current_project.get("id")
        url = f"{self.config.api_base_url}/projects/{project_id}/data?task={task_id}&labeling=1"

        self.console.print(
            f"[green]Opening Label Studio interface in browser...[/green]"
        )
        self.console.print(f"[cyan]URL: {url}[/cyan]")

        try:
            webbrowser.open(url)
            self.console.print("[green]✓ Browser opened successfully[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to open browser: {e}[/red]")
            self.console.print(f"[yellow]Please manually open: {url}[/yellow]")

    def launch_sklearn_text_classifier_interface(self):
        """Launch specialized labeling interface for sklearn text classifier."""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        project_id = self.current_project.get("id")

        # Get project tasks to check if any exist
        tasks = self.client.get_tasks(project_id)
        if not tasks or not tasks.get("tasks"):
            self.console.print(
                "[yellow]No tasks found. Please import tasks first.[/yellow]"
            )
            return

        # Launch options
        launch_options = {
            "1": "Open sklearn text classifier interface in browser",
            "2": "Generate custom sklearn frontend HTML",
        }

        self.console.print("\n[bold]Sklearn Text Classifier Interface Options:[/bold]")
        for key, option in launch_options.items():
            self.console.print(f"{key}. {option}")

        choice = Prompt.ask(
            "Choose labeling interface",
            choices=list(launch_options.keys()),
            default="2",
        )

        if choice == "1":
            self.open_browser_interface(
                tasks["tasks"][0]["id"] if tasks.get("tasks") else {}
            )
        elif choice == "2":
            self.generate_sklearn_frontend()

    def generate_sklearn_frontend(self):
        """Generate a custom HTML frontend for sklearn text classification."""

        if not self.current_project:
            self.console.print("[red]Failed to get project details[/red]")
            return

        # Get first task for demo
        tasks = self.client.get_tasks(self.current_project.get("id"))
        sample_task = tasks["tasks"][0] if tasks.get("tasks") else {}

        html_content = self.create_sklearn_frontend_html(sample_task)

        filename = (
            f"sklearn_text_classifier_project_{self.current_project.get('id')}.html"
        )

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_content)

            self.console.print(
                f"[green]Sklearn text classifier frontend generated: {filename}[/green]"
            )

            if Confirm.ask("Open the generated HTML file?", default=True):
                import os
                import webbrowser

                webbrowser.open(f"file://{os.path.abspath(filename)}")

        except Exception as e:
            self.console.print(f"[red]Failed to generate frontend: {e}[/red]")

    def create_sklearn_frontend_html(self, sample_task):
        """Create the HTML content for the sklearn text classifier frontend."""
        project_title = self.current_project.get("title", "Sklearn Text Classification")
        self.current_project.get("label_config")
        # Enhanced HTML template for sklearn text classification
        html_template = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{project_title} - Sklearn Text Classifier</title>
        
        <!-- Label Studio Frontend CSS -->
        <link href="https://unpkg.com/@heartexlabs/label-studio@latest/build/static/css/main.css" rel="stylesheet">
        
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            
            .header {{
                background: rgba(255, 255, 255, 0.95);
                color: #2c3e50;
                padding: 25px;
                border-radius: 12px;
                margin-bottom: 20px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
            }}
            
            .sklearn-badge {{
                display: inline-block;
                background: linear-gradient(45deg, #f39c12, #e67e22);
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                margin-left: 10px;
            }}
            
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.98);
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                overflow: hidden;
                backdrop-filter: blur(10px);
            }}
            
            .toolbar {{
                background: linear-gradient(135deg, #2c3e50, #34495e);
                color: white;
                padding: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .sentiment-indicators {{
                display: flex;
                gap: 15px;
                margin-bottom: 20px;
                justify-content: center;
            }}
            
            .sentiment-card {{
                padding: 15px 25px;
                border-radius: 25px;
                font-weight: bold;
                color: white;
                text-align: center;
                min-width: 120px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                transition: transform 0.3s ease;
            }}
            
            .sentiment-card:hover {{
                transform: translateY(-2px);
            }}
            
            .positive-card {{
                background: linear-gradient(135deg, #27ae60, #2ecc71);
            }}
            
            .negative-card {{
                background: linear-gradient(135deg, #e74c3c, #c0392b);
            }}
            
            .neutral-card {{
                background: linear-gradient(135deg, #95a5a6, #7f8c8d);
            }}
            
            .btn {{
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .btn-primary {{
                background: linear-gradient(135deg, #3498db, #2980b9);
                color: white;
            }}
            
            .btn-primary:hover {{
                background: linear-gradient(135deg, #2980b9, #1f4e79);
                transform: translateY(-1px);
            }}
            
            .btn-success {{
                background: linear-gradient(135deg, #27ae60, #229954);
                color: white;
            }}
            
            .btn-success:hover {{
                background: linear-gradient(135deg, #229954, #1e7e34);
                transform: translateY(-1px);
            }}
            
            .btn-warning {{
                background: linear-gradient(135deg, #f39c12, #e67e22);
                color: white;
            }}
            
            .btn-warning:hover {{
                background: linear-gradient(135deg, #e67e22, #d35400);
                transform: translateY(-1px);
            }}
            
            #label-studio {{
                min-height: 500px;
                padding: 30px;
                background: #fafafa;
            }}
            
            .task-info {{
                background: linear-gradient(135deg, #ecf0f1, #bdc3c7);
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 8px;
                font-size: 14px;
                border-left: 4px solid #3498db;
            }}
            
            .status {{
                padding: 6px 12px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: 600;
            }}
            
            .status-ready {{
                background: linear-gradient(135deg, #d4edda, #c3e6cb);
                color: #155724;
            }}
            
            .sklearn-info {{
                background: linear-gradient(135deg, #fff3cd, #ffeaa7);
                border: 1px solid #ffeaa7;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            }}
            
            .prediction-confidence {{
                display: flex;
                justify-content: space-between;
                margin-top: 10px;
                                  }}
            
            .confidence-bar {{
                height: 6px;
                background: #ecf0f1;
                border-radius: 3px;
                overflow: hidden;
                margin-top: 5px;
            }}
            
            .confidence-fill {{
                height: 100%;
                background: linear-gradient(90deg, #e74c3c, #f39c12, #27ae60);
                transition: width 0.3s ease;
            }}
            
            .hotkey-hint {{
                font-size: 11px;
                color: #7f8c8d;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 {project_title}</h1>
            <span class="sklearn-badge">SKLEARN POWERED</span>
            <p>Intelligent Text Sentiment Classification</p>
                💡 <strong>Quick Tips:</strong> Use P for Positive, N for Negative, Ctrl+Enter to Submit, Ctrl+S to Skip
            </div>
            
            <!-- Label Studio Frontend Container -->
            <div id="label-studio"></div>
        </div>

        <!-- Label Studio Frontend JavaScript -->
        <script src="https://unpkg.com/@heartexlabs/label-studio@latest/build/static/js/main.js"></script>
        
        <script>
            // Configuration and task data
            const API_BASE_URL = '{self.config.api_base_url}';
            const API_KEY = '{self.config.api_key}';
            const PROJECT_ID = {self.current_project.get('id')};
            
            let currentTaskIndex = 0;
            let tasks = [];
            let labelStudio = null;
            
            // Sklearn-optimized Label Studio configuration
            const labelStudioConfig = `{self.current_project.get('label_config')}`;
            
            // Sample task data
            const sampleTask = {json.dumps(sample_task, indent=8)};
            
            // Initialize Label Studio
            function initializeLabelStudio(taskData) {{
                if (labelStudio) {{
                    labelStudio.destroy();
                }}
                
                labelStudio = new LabelStudio('label-studio', {{
                    config: labelStudioConfig,
                    
                    interfaces: [
                        "panel",
                        "update", 
                        "controls",
                        "side-column",
                        "annotations:menu",
                        "annotations:add-new",
                        "annotations:delete",
                        "predictions:menu",
                        "auto-annotation"
                    ],
                    
                    user: {{
                        pk: 1,
                        firstName: "Sklearn",
                        lastName: "Annotator"
                    }},
                    
                    task: taskData || sampleTask,
                    
                    onLabelStudioLoad: function(LS) {{
                        console.log('Sklearn Label Studio loaded successfully');
                        updateTaskInfo();
                        simulateSklearnPrediction();
                    }},
                    
                    onSubmitAnnotation: function(LS, annotation) {{
                        console.log('Annotation submitted:', annotation);
                        submitAnnotation();
                    }},
                    
                    onUpdateAnnotation: function(LS, annotation) {{
                        console.log('Annotation updated:', annotation);
                        updateConfidenceDisplay(annotation);
                    }},
                    
                    onSkipTask: function(LS) {{
                        console.log('Task skipped');
                        skipTask();
                    }},
                    
                    onTaskLoad: function(LS, task) {{
                        console.log('Task loaded:', task);
                        updateTaskInfo();
                        simulateSklearnPrediction();
                    }}
                }});
            }}
            
            // Simulate sklearn model prediction
            function simulateSklearnPrediction() {{
                const currentTask = getCurrentTask();
                if (currentTask && currentTask.text) {{
                    // Simple sentiment analysis simulation
                    const text = currentTask.text.toLowerCase();
                    const positiveWords = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'wonderful', 'fantastic'];
                    const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst', 'disappointing'];
                    
                    let positiveScore = 0;
                    let negativeScore = 0;
                    
                    positiveWords.forEach(word => {{
                        if (text.includes(word)) positiveScore += 1;
                    }});
                    
                    negativeWords.forEach(word => {{
                        if (text.includes(word)) negativeScore += 1;
                    }});
                    
                    const totalScore = positiveScore + negativeScore;
                    let confidence = 0.5; // Default neutral
                    let prediction = 'Neutral';
                    
                    if (totalScore > 0) {{
                        confidence = Math.max(positiveScore, negativeScore) / totalScore;
                        prediction = positiveScore > negativeScore ? 'Positive' : 'Negative';
                        confidence = Math.min(0.95, 0.6 + (confidence * 0.35)); // Scale to 60-95%
                    }} else {{
                        confidence = 0.5 + (Math.random() * 0.3); // Random confidence for neutral
                        prediction = Math.random() > 0.5 ? 'Positive' : 'Negative';
                    }}
                    
                    updateConfidenceDisplay(null, confidence, prediction);
                }}
            }}
            
            // Update confidence display
            function updateConfidenceDisplay(annotation, confidence = null, prediction = null) {{
                if (confidence !== null) {{
                    const confidencePercent = Math.round(confidence * 100);
                    document.getElementById('confidence-score').textContent = `${{confidencePercent}}%`;
                    document.getElementById('confidence-fill').style.width = `${{confidencePercent}}%`;
                    
                    // Update confidence bar color based on prediction
                    const fillElement = document.getElementById('confidence-fill');
                    if (prediction === 'Positive') {{
                        fillElement.style.background = 'linear-gradient(90deg, #27ae60, #2ecc71)';
                    }} else if (prediction === 'Negative') {{
                        fillElement.style.background = 'linear-gradient(90deg, #e74c3c, #c0392b)';
                    }} else {{
                        fillElement.style.background = 'linear-gradient(90deg, #95a5a6, #7f8c8d)';
                    }}
                }}
            }}
            
            // API Functions
            async function fetchTasks() {{
                try {{
                    const response = await fetch(`${{API_BASE_URL}}/api/tasks/?project=${{PROJECT_ID}}`, {{
                        headers: {{
                            'Authorization': `Token ${{API_KEY}}`,
                            'Content-Type': 'application/json'
                        }}
                    }});
                    
                    if (response.ok) {{
                        const data = await response.json();
                        tasks = data.tasks || [];
                        updateTaskInfo();
                        return tasks;
                    }} else {{
                        console.error('Failed to fetch tasks:', response.statusText);
                        return [];
                    }}
                }} catch (error) {{
                    console.error('Error fetching tasks:', error);
                    return [];
                }}
            }}
            
            async function submitAnnotation() {{
                if (!labelStudio) return;
                
                const annotations = labelStudio.annotationStore.annotations;
                const currentAnnotation = annotations[annotations.length - 1];
                
                if (!currentAnnotation) {{
                    alert('No annotation to submit');
                    return;
                }}
                
                try {{
                    const response = await fetch(`${{API_BASE_URL}}/api/tasks/${{getCurrentTask().id}}/annotations/`, {{
                        method: 'POST',
                        headers: {{
                            'Authorization': `Token ${{API_KEY}}`,
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{
                            result: currentAnnotation.serializeAnnotation(),
                            task: getCurrentTask().id
                        }})
                    }});
                    
                    if (response.ok) {{
                        // Show success animation
                        showSuccessAnimation();
                        setTimeout(() => {{
                            loadNextTask();
                        }}, 1000);
                    }} else {{
                        alert('Failed to submit annotation');
                    }}
                }} catch (error) {{
                    console.error('Error submitting annotation:', error);
                    alert('Error submitting annotation');
                }}
            }}
            
            function showSuccessAnimation() {{
                const toolbar = document.querySelector('.toolbar');
                toolbar.style.background = 'linear-gradient(135deg, #27ae60, #2ecc71)';
                setTimeout(() => {{
                    toolbar.style.background = 'linear-gradient(135deg, #2c3e50, #34495e)';
                }}, 1000);
            }}
            
            function loadNextTask() {{
                if (currentTaskIndex < tasks.length - 1) {{
                    currentTaskIndex++;
                    loadCurrentTask();
                }} else {{
                    alert('🎉 All tasks completed! Great job!');
                }}
            }}
            
            function loadPreviousTask() {{
                if (currentTaskIndex > 0) {{
                    currentTaskIndex--;
                    loadCurrentTask();
                }} else {{
                    alert('This is the first task');
                }}
            }}
            
            function loadCurrentTask() {{
                const task = getCurrentTask();
                if (task) {{
                    initializeLabelStudio(task);
                }}
            }}
            
            function skipTask() {{
                loadNextTask();
            }}
            
            function updateTaskInfo() {{
                document.getElementById('current-task-id').textContent = getCurrentTask().id || 'Demo';
                document.getElementById('task-progress').textContent = `${{currentTaskIndex + 1}}/${{Math.max(tasks.length, 1)}}`;
                document.getElementById('task-status').textContent = tasks.length > 0 ? 'Loaded' : 'Demo Mode';
            }}
            
            // Initialize the application
            window.addEventListener('DOMContentLoaded', function() {{
                console.log('Initializing Sklearn Text Classifier Frontend...');
                
                // Load tasks and initialize
                fetchTasks().then(loadedTasks => {{
                    if (loadedTasks.length > 0) {{
                        initializeLabelStudio(loadedTasks[0]);
                    }} else {{
                        // Use sample task for demo
                        initializeLabelStudio(sampleTask);
                    }}
                }});
            }});
            
            // Enhanced keyboard shortcuts for sklearn
            document.addEventListener('keydown', function(e) {{
                if (e.ctrlKey || e.metaKey) {{
                    switch(e.key) {{
                        case 'ArrowLeft':
                            e.preventDefault();
                            loadPreviousTask();
                            break;
                        case 'ArrowRight':
                            e.preventDefault();
                            loadNextTask();
                            break;
                        case 'Enter':
                            e.preventDefault();
                            submitAnnotation();
                            break;
                        case 's':
                            e.preventDefault();
                            skipTask();
                            break;
                    }}
                }} else {{
                    // Quick sentiment selection
                    switch(e.key.toLowerCase()) {{
                        case 'p':
                            // Trigger positive selection
                            if (labelStudio) {{
                                const positiveChoice = document.querySelector('[data-value="Positive"]');
                                if (positiveChoice) positiveChoice.click();
                            }}
                            break;
                        case 'n':
                            // Trigger negative selection
                            if (labelStudio) {{
                                const negativeChoice = document.querySelector('[data-value="Negative"]');
                                if (negativeChoice) negativeChoice.click();
                            }}
                            break;
                    }}
                }}
            }});
        </script>
    </body>
    </html>"""

        return html_template

    def select_frontend_template(self) -> str:
        """Interactive template selection."""
        templates = self.template_manager.list_templates()

        if not templates:
            self.console.print(
                "[yellow]No templates found. Creating default templates...[/yellow]"
            )
            return

        self.console.print("[bold blue]Available Frontend Templates:[/bold blue]")
        for i, template in enumerate(templates, 1):
            self.console.print(f"{i}. {template}")

        while True:
            try:
                choice = int(Prompt.ask("Select template number")) - 1
                if 0 <= choice < len(templates):
                    return templates[choice]
                else:
                    self.console.print("[red]Invalid selection[/red]")
            except ValueError:
                self.console.print("[red]Please enter a valid number[/red]")

    def launch_enhanced_labeling_interface(self):
        """Launch labeling interface with template selection and auto-labeling."""
        if not self.current_project:
            self.console.print("[red]No project selected[/red]")
            return

        # Select template
        template_name = self.select_frontend_template()
        template_content = self.template_manager.load_template(template_name)

        # Get auto-labeling parameters
        auto_label_count = int(
            Prompt.ask("Number of tasks to auto-label", default="10")
        )
        batch_size = int(Prompt.ask("Batch size for processing", default="5"))

        # Launch interface with auto-labeling
        self._launch_interface_with_auto_labeling(
            template_content, template_name, auto_label_count, batch_size
        )

    def _launch_interface_with_auto_labeling(
        self,
        template_content: str,
        template_name: str,
        auto_label_count: int,
        batch_size: int,
    ):
        """Launch interface and perform auto-labeling."""
        project_id = self.current_project["id"]

        # Get tasks for auto-labeling
        tasks = self.client.get_tasks(project_id)
        # for task in tasks["tasks"]:
        #     if not task.get("annotations"):  # Prüft, ob keine Annotationen vorhanden sind
        #         tasks_without_annotations.append(task)

        unlabeled_tasks = [
            task for task in tasks["tasks"] if not task.get("annotations")
        ]

        if len(unlabeled_tasks) < auto_label_count:
            self.console.print(
                f"[yellow]Only {len(unlabeled_tasks)} unlabeled tasks available[/yellow]"
            )
            auto_label_count = len(unlabeled_tasks)

        # Process tasks in batches
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task_progress = progress.add_task(
                f"Auto-labeling with {template_name} template...",
                total=auto_label_count,
            )

            for i in range(0, auto_label_count, batch_size):
                batch = unlabeled_tasks[i : i + batch_size]
                self._process_batch_with_template(
                    batch, template_content, template_name
                )
                progress.update(task_progress, advance=len(batch))

        self.console.print(
            f"[green]Auto-labeled {auto_label_count} tasks using {template_name} template[/green]"
        )

        # Launch the labeling interface
        self._open_labeling_interface(template_content, template_name)

    def _process_batch_with_template(
        self, tasks: List[Dict], template_content: str, template_name: str
    ):
        """Process a batch of tasks with the selected template."""
        for task in tasks:
            try:
                # Generate predictions based on template type
                if template_name == "sklearn_classifier":
                    prediction = self._generate_sklearn_prediction(task)
                elif template_name == "gait_analysis":
                    prediction = self._generate_gait_prediction(task)
                else:
                    prediction = self._generate_basic_prediction(task)

                if prediction:
                    self.client.create_prediction(
                        task_id=task["id"],
                        result=prediction["result"],
                        score=prediction.get("score", 0.8),
                    )
            except Exception as e:
                self.logger.error(f"Failed to process task {task['id']}: {e}")

    def _generate_sklearn_prediction(self, task: Dict) -> Optional[Dict]:
        """Generate prediction for sklearn classifier template."""
        # Implement your sklearn-based prediction logic here
        text_data = task.get("data", {}).get("text", "")

        # Example prediction structure for text classification
        prediction = {
            "result": [
                {
                    "value": {"choices": ["positive"]},  # or "negative", "neutral"
                    "from_name": "sentiment",
                    "to_name": "text",
                    "type": "choices",
                }
            ],
            "score": 0.85,
        }
        return prediction

    def _open_labeling_interface(self, template_content: str, template_name: str):
        """Open the labeling interface with the given template content and name."""
        import os
        import tempfile
        import webbrowser

        try:
            # Create a temporary HTML file with the template content
            with tempfile.NamedTemporaryFile(
                "w", delete=False, suffix=".html", encoding="utf-8"
            ) as f:
                f.write(template_content)
                temp_file_path = f.name

            # Open the temporary file in the default web browser
            file_url = f"file://{os.path.abspath(temp_file_path)}"
            webbrowser.open(file_url)

            self.console.print(
                f"[green]Opened labeling interface with template: {template_name}[/green]"
            )
            self.console.print(f"[blue]Interface URL: {file_url}[/blue]")

            # Store temp file path for cleanup if needed
            if not hasattr(self, "_temp_files"):
                self._temp_files = []
            self._temp_files.append(temp_file_path)

        except Exception as e:
            self.console.print(f"[red]Failed to open labeling interface: {e}[/red]")

    def setup_active_learning_project(self):
        """Setup a project specifically for active learning."""
        if not self.client:
            self.console.print("[red]No client initialized[/red]")
            return

        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        from .active_learning import ActiveLearningManager, WebhookHandler

        # Get project selection
        project_id = self.current_project.get("id")

        # Configure active learning parameters
        batch_size = int(
            Prompt.ask("Enter batch size for active learning", default="10")
        )
        uncertainty_strategy = Prompt.ask(
            "Select uncertainty strategy",
            choices=["entropy", "least_confidence", "margin"],
            default="entropy",
        )

        # Setup active learning manager
        self.al_manager = ActiveLearningManager(
            api_client=self.client,
            project_id=project_id,
            uncertainty_strategy=uncertainty_strategy,
            batch_size=batch_size,
        )

        # Setup webhook for automation
        webhook_url = Prompt.ask(
            "Enter webhook URL", default="http://localhost:5000/webhook"
        )
        webhook_result = self.client.create_webhook(
            project_id=project_id,
            url=webhook_url,
            events=["ANNOTATION_CREATED", "ANNOTATION_UPDATED"],
        )

        if webhook_result:
            self.console.print("[green]Active learning setup complete![/green]")
            self.webhook_handler = WebhookHandler(self.al_manager)
        else:
            self.console.print(
                "[yellow]Warning: Webhook setup failed, manual triggering required[/yellow]"
            )

    def start_active_learning_session(self):
        """Start an interactive active learning session."""
        if not self.current_project:
            self.console.print("[yellow]No project selected[/yellow]")
            return

        from .active_learning import ActiveLearningManager

        project_id = self.current_project.get("id")

        # Check prerequisites
        ml_backends = self.client.list_ml_backends(project_id)
        if not ml_backends:
            self.console.print(
                "[yellow]No ML backends found. Please add one first.[/yellow]"
            )
            if Confirm.ask("Add ML backend now?", default=True):
                self.add_ml_backend()
                return

        # Get active learning parameters
        self.console.print("[bold blue]Active Learning Session Setup[/bold blue]")

        batch_size = int(Prompt.ask("Batch size for each iteration", default="3"))
        max_iterations = int(Prompt.ask("Maximum iterations", default="2"))
        min_confidence = float(
            Prompt.ask("Minimum uncertainty threshold (0.0-1.0)", default="0.05")
        )

        strategies = ["entropy", "least_confidence", "margin"]
        self.console.print("\n[bold]Uncertainty Sampling Strategies:[/bold]")
        for i, strategy in enumerate(strategies, 1):
            self.console.print(f"{i}. {strategy}")

        strategy_choice = int(
            Prompt.ask("Choose strategy", choices=["1", "2", "3"], default="1")
        )
        uncertainty_strategy = strategies[strategy_choice - 1]

        # Initialize active learning manager
        al_manager = ActiveLearningManager(
            api_client=self.client,
            project_id=project_id,
            uncertainty_strategy=uncertainty_strategy,
            batch_size=batch_size,
            min_confidence_threshold=min_confidence,
        )

        # Run active learning loop
        self.console.print(f"\n[green]Starting Active Learning Session[/green]")
        self.console.print(
            f"Project: {self.current_project.get('title')} (ID: {project_id})"
        )

        for iteration in range(max_iterations):
            self.console.print(
                f"\n[bold cyan]--- Iteration {iteration + 1}/{max_iterations} ---[/bold cyan]"
            )

            # Select next batch
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    "Selecting high-uncertainty tasks...", total=None
                )

                selected_tasks = al_manager.select_next_batch()

                if not selected_tasks:
                    progress.update(
                        task,
                        description="[yellow]No more uncertain tasks available[/yellow]",
                    )
                    progress.stop()
                    self.console.print(
                        "[yellow]No tasks available for annotation. Session complete![/yellow]"
                    )
                    break

                progress.update(
                    task,
                    description=f"[green]Selected {len(selected_tasks)} high-uncertainty tasks[/green]",
                )
                progress.stop()

            # Display selected tasks
            self._display_active_learning_batch(selected_tasks, al_manager)

            # Launch annotation interface
            if Confirm.ask("Open Label Studio interface for annotation?", default=True):
                self._launch_active_learning_interface(selected_tasks)

            # Wait for user to complete annotations
            self.console.print("[yellow]Please annotate the selected tasks...[/yellow]")
            input("Press Enter when annotations are complete...")

            # Update labeled tasks and trigger retraining
            self.console.print("[blue]Triggering model retraining...[/blue]")
            al_manager.trigger_retraining(project_id, ml_backends[0]["id"])
            
            # Show progress
            stats = al_manager.get_learning_stats()
            self.console.print(
                f"[green]Progress: {stats['labeled_tasks']} tasks labeled[/green]"
            )

            # Ask if user wants to continue
            if iteration < max_iterations - 1:
                if not Confirm.ask("Continue to next iteration?", default=True):
                    break

        # Final summary
        final_stats = al_manager.get_learning_stats()
        self.console.print(
            Panel(
                f"[green]Active Learning Session Complete![/green]\n\n"
                f"• Total iterations: {final_stats['current_iteration']}\n"
                f"• Tasks labeled: {final_stats['labeled_tasks']}\n"
                f"• Strategy used: {final_stats['uncertainty_strategy']}\n"
                f"• Batch size: {final_stats['batch_size']}",
                title="Session Summary",
                border_style="green",
            )
        )

    def _display_active_learning_batch(self, task_ids: list, al_manager):
        """Display the selected batch for active learning."""
        table = Table(title="🎯 Active Learning Batch Selection")
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Uncertainty Level", style="yellow")

        for task_id in task_ids:
            table.add_row(str(task_id), "Ready for annotation", "🔴 High")

        self.console.print(table)

        # Show batch statistics
        stats = al_manager.get_learning_stats()
        self.console.print(
            f"[blue]Batch {stats['current_iteration'] + 1} | "
            f"Strategy: {stats['uncertainty_strategy']} | "
            f"Total labeled: {stats['labeled_tasks']}[/blue]"
        )

    def _launch_active_learning_interface(self, task_ids: list):
        """Launch Label Studio interface filtered to show only selected tasks."""
        import webbrowser

        project_id = self.current_project.get("id")
        base_url = self.config.api_base_url

        if not task_ids:
            # Fallback to project data page
            url = f"{base_url}/projects/{project_id}/data?tab=0"
            self.console.print(
                f"[blue]Opening Label Studio project data page...[/blue]"
            )
            try:
                webbrowser.open(url)
                self.console.print(f"[green]✓ Interface opened: {url}[/green]")
            except Exception as e:
                self.console.print(f"[red]Failed to open browser: {e}[/red]")
                self.console.print(f"[yellow]Please manually open: {url}[/yellow]")
            return

        # Ask user if they want to open all tasks in separate tabs
        num_tasks = len(task_ids)
        # if num_tasks > 1:
        #     self.console.print(f"[yellow]📋 {num_tasks} tasks selected for annotation[/yellow]")
        #     # open_all = Confirm.ask(
        #     #     f"Open all {num_tasks} tasks in separate browser tabs?",
        #     #     default=True
        #     # )
        # else:
        #     open_all = True
        #     self.console.print(f"[yellow]📋 Opening task {task_ids[0]} for annotation[/yellow]")

        open_all = True
        if open_all:
            self.console.print(
                f"[blue]Opening {num_tasks} task(s) in separate browser tabs...[/blue]"
            )

            # Import time for delays between tab opening
            import time

            success_count = 0
            for i, task_id in enumerate(task_ids):
                url = f"{base_url}/projects/{project_id}/data?task={task_id}&tab=0"

                try:
                    # Add a small delay between opening tabs to avoid overwhelming the browser
                    if i > 0:
                        time.sleep(0.5)

                    # For Firefox, we can use a specific command to open in new tabs
                    # Try using firefox directly first, then fall back to webbrowser
                    import subprocess

                    try:
                        # Try to use firefox command with new tab option
                        subprocess.run(
                            ["firefox", "--new-tab", url],
                            check=False,
                            capture_output=True,
                            timeout=2,
                        )
                        success_count += 1
                        self.console.print(
                            f"[dim]  ✓ Task {task_id} opened in Firefox[/dim]"
                        )
                    except (
                        subprocess.TimeoutExpired,
                        subprocess.CalledProcessError,
                        FileNotFoundError,
                    ):
                        # Fall back to webbrowser
                        webbrowser.open_new_tab(url)
                        success_count += 1
                        self.console.print(
                            f"[dim]  ✓ Task {task_id} opened in browser[/dim]"
                        )

                except Exception as e:
                    self.console.print(
                        f"[red]  ✗ Failed to open task {task_id}: {e}[/red]"
                    )
                    self.console.print(f"[yellow]    Manual URL: {url}[/yellow]")

            if success_count > 0:
                self.console.print(
                    f"[green]✓ Successfully opened {success_count}/{num_tasks} tasks[/green]"
                )
                if success_count == num_tasks:
                    self.console.print(
                        f"[dim]💡 Each task is now open in its own browser tab for easy annotation[/dim]"
                    )
            else:
                self.console.print(f"[red]Failed to open any tasks automatically[/red]")
        else:
            # Fall back to opening just the first task
            first_task_id = task_ids[0]
            url = f"{base_url}/projects/{project_id}/data?task={first_task_id}&tab=0"

            self.console.print(f"[blue]Opening first task only...[/blue]")
            self.console.print(
                f"[yellow]📋 Remaining Task IDs: {', '.join(map(str, task_ids))}[/yellow]"
            )

            try:
                webbrowser.open(url)
                self.console.print(f"[green]✓ Interface opened: {url}[/green]")
            except Exception as e:
                self.console.print(f"[red]Failed to open browser: {e}[/red]")
                self.console.print(f"[yellow]Please manually open: {url}[/yellow]")

            # Wait for annotations (in real implementation, this would be event-driven)
            self.console.print(
                "[yellow]Waiting for annotations to complete...[/yellow]"
            )
            input("Press Enter when annotation batch is complete...")

            # Trigger retraining
            self.al_manager.trigger_model_retraining()

            # Wait for training to complete
            self.console.print("[yellow]Model retraining in progress...[/yellow]")
            input("Press Enter when model training is complete...")

    def get_active_learning_metrics(self):
        """Get metrics about active learning performance."""
        if not hasattr(self, "al_manager"):
            return {}

        return {
            "total_labeled": len(self.al_manager.labeled_task_ids),
            "current_iteration": self.al_manager.current_iteration,
            "batch_size": self.al_manager.batch_size,
            "uncertainty_strategy": self.al_manager.uncertainty_sampler.strategy,
            "diversity_strategy": self.al_manager.diversity_sampler.strategy,
        }

    def view_active_learning_status(self):
        """
        Display comprehensive active learning status for the current project.

        This function provides detailed insights into:
        - Active learning progress and effectiveness
        - Model training status and performance
        - Task selection and sampling metrics
        - Annotation progress and quality indicators
        """
        if not self.current_project:
            self.console.print(
                "[red]No project selected. Please select a project first.[/red]"
            )
            return

        project_id = self.current_project["id"]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:

            # Gather comprehensive status information
            status_task = progress.add_task(
                "Gathering active learning status...", total=None
            )

            # Get basic project statistics
            project_stats = self.client.get_project_statistics(project_id)

            # Get ML backend information
            ml_backends = self.client.list_ml_backends(project_id)

            # Get recent predictions for uncertainty analysis
            predictions = self.client.list_predictions(project_id=project_id)

            # Get recent tasks for sampling analysis
            tasks = self.client.get_tasks(project_id)

            progress.update(
                status_task, description="[green]Status information collected[/green]"
            )
            progress.stop()

        # Create comprehensive status display
        self._display_active_learning_dashboard(
            project_stats, ml_backends, predictions, tasks
        )

    def _display_active_learning_dashboard(
        self, project_stats, ml_backends, predictions, tasks
    ):
        """
        Display a comprehensive active learning status dashboard.
        """

        # Main dashboard panel
        self.console.print(
            Panel.fit(
                f"[bold blue]Active Learning Status Dashboard[/bold blue]\n"
                f"Project: {self.current_project.get('title', 'Unknown')}\n"
                f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                title="Active Learning Monitor",
            )
        )

        # Project Progress Overview
        self._display_project_progress(project_stats)

        # ML Backend Status
        self._display_ml_backend_status(ml_backends)

        # Active Learning Effectiveness Metrics
        self._display_active_learning_metrics(predictions, tasks, project_stats)

        # Sampling Strategy Performance
        self._display_sampling_performance(predictions, tasks)

        # Recommendations and Next Steps
        self._display_recommendations(project_stats, ml_backends, predictions)

    def _display_project_progress(self, stats):
        """Display project progress metrics."""
        if not stats:
            self.console.print("[yellow]No project statistics available[/yellow]")
            return

        table = Table(title="Project Progress Overview")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")

        total_tasks = stats.get("total_tasks", 0)
        annotated_tasks = stats.get("annotated_tasks", 0)
        completion_rate = stats.get("completion_rate", 0)

        # Progress indicators
        progress_status = (
            "🟢 On Track"
            if completion_rate > 50
            else "🟡 In Progress" if completion_rate > 20 else "🔴 Starting"
        )

        table.add_row("Total Tasks", str(total_tasks), f"{total_tasks} tasks loaded")
        table.add_row(
            "Annotated Tasks", str(annotated_tasks), f"{annotated_tasks} completed"
        )
        table.add_row("Completion Rate", f"{completion_rate:.1f}%", progress_status)
        table.add_row(
            "Remaining Tasks",
            str(stats.get("remaining_tasks", 0)),
            f"{stats.get('remaining_tasks', 0)} pending",
        )

        self.console.print(table)

    def _display_ml_backend_status(self, ml_backends):
        """Display ML backend status and training information."""
        if not ml_backends:
            self.console.print("[yellow]No ML backends configured[/yellow]")
            return

        table = Table(title="ML Backend Status")
        table.add_column("Backend ID", style="cyan")
        table.add_column("URL", style="blue")
        table.add_column("Status", style="green")
        table.add_column("Last Training", style="yellow")

        for backend in ml_backends:
            backend_id = backend.get("id", "Unknown")

            # Get training status for each backend
            training_status = self.client.get_training_status(backend_id)
            status_text = "🟢 Ready" if training_status else "🟡 Unknown"

            table.add_row(
                str(backend_id),
                backend.get("url", "Unknown")[:50],
                status_text,
                "Recently" if training_status else "Unknown",
            )

        self.console.print(table)

    def _display_active_learning_metrics(self, predictions, tasks, project_stats):
        """Display active learning effectiveness metrics."""
        if not predictions:
            self.console.print("[yellow]No predictions available for analysis[/yellow]")
            return

        # Calculate uncertainty distribution using proper prediction structure
        prediction_scores = []
        for pred in predictions:
            if isinstance(pred, dict):
                # Get score from main prediction
                score = pred.get("score")

                # If no main score, try to get from result items
                if score is None and pred.get("result"):
                    for result_item in pred.get("result", []):
                        if isinstance(result_item, dict) and "score" in result_item:
                            score = result_item["score"]
                            break

                if score is not None:
                    prediction_scores.append(score)

        if not prediction_scores:
            self.console.print(
                "[yellow]No prediction scores available for analysis[/yellow]"
            )
            return

        # Calculate metrics
        avg_confidence = sum(prediction_scores) / len(prediction_scores)
        high_uncertainty_count = sum(1 for score in prediction_scores if score < 0.7)
        very_uncertain_count = sum(1 for score in prediction_scores if score < 0.5)

        # Calculate uncertainty distribution
        low_conf = sum(1 for score in prediction_scores if score < 0.5)
        med_conf = sum(1 for score in prediction_scores if 0.5 <= score < 0.8)
        high_conf = sum(1 for score in prediction_scores if score >= 0.8)

        table = Table(title="Active Learning Effectiveness")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Interpretation", style="green")

        table.add_row(
            "Average Prediction Confidence",
            f"{avg_confidence:.3f}",
            "Higher is better" if avg_confidence > 0.8 else "Model improving",
        )

        table.add_row(
            "High Uncertainty Tasks (< 0.7)",
            str(high_uncertainty_count),
            (
                "Good targets for annotation"
                if high_uncertainty_count > 5
                else "Few uncertain samples"
            ),
        )

        table.add_row(
            "Very Uncertain Tasks (< 0.5)",
            str(very_uncertain_count),
            (
                "Prime annotation targets"
                if very_uncertain_count > 0
                else "Model quite confident"
            ),
        )

        table.add_row(
            "Prediction Coverage",
            f"{len(predictions)}/{project_stats.get('total_tasks', 1) if project_stats else len(predictions)}",
            "Tasks with predictions",
        )

        # Add confidence distribution
        table.add_row("", "", "")  # Spacer
        table.add_row(
            "Low Confidence (< 0.5)",
            f"{low_conf} ({low_conf/len(prediction_scores):.1%})",
            "🎯 Prime for AL",
        )
        table.add_row(
            "Medium Confidence (0.5-0.8)",
            f"{med_conf} ({med_conf/len(prediction_scores):.1%})",
            "⚖️ Moderate targets",
        )
        table.add_row(
            "High Confidence (≥ 0.8)",
            f"{high_conf} ({high_conf/len(prediction_scores):.1%})",
            "✅ Model confident",
        )

        self.console.print(table)

    def _display_sampling_performance(self, predictions, tasks):
        """Display sampling strategy performance analysis."""

        # Analyze prediction score distribution with correct structure
        if predictions:
            scores = []
            for pred in predictions:
                if isinstance(pred, dict):
                    score = pred.get("score")
                    # Fallback to result item score if main score not available
                    if score is None and pred.get("result"):
                        for result_item in pred.get("result", []):
                            if isinstance(result_item, dict) and "score" in result_item:
                                score = result_item["score"]
                                break
                    if score is not None:
                        scores.append(score)

            if scores:
                low_conf = sum(1 for s in scores if s < 0.5)
                med_conf = sum(1 for s in scores if 0.5 <= s < 0.8)
                high_conf = sum(1 for s in scores if s >= 0.8)

                # Calculate additional metrics
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)

                table = Table(title="Sampling Strategy Performance")
                table.add_column("Confidence Range", style="cyan")
                table.add_column("Count", style="magenta")
                table.add_column("Percentage", style="yellow")
                table.add_column("Strategy Impact", style="green")

                total = len(scores)
                table.add_row(
                    "Low (< 0.5)",
                    str(low_conf),
                    f"{low_conf/total:.1%}",
                    "🎯 Highest AL value",
                )
                table.add_row(
                    "Medium (0.5-0.8)",
                    str(med_conf),
                    f"{med_conf/total:.1%}",
                    "⚖️ Moderate AL value",
                )
                table.add_row(
                    "High (≥ 0.8)",
                    str(high_conf),
                    f"{high_conf/total:.1%}",
                    "✅ Low AL value",
                )

                # Add summary statistics
                table.add_row("", "", "", "")  # Spacer
                table.add_row(
                    "Average Score",
                    f"{avg_score:.3f}",
                    "Overall confidence",
                    "Model performance indicator",
                )
                table.add_row(
                    "Score Range",
                    f"{min_score:.3f} - {max_score:.3f}",
                    "Confidence spread",
                    "Uncertainty diversity",
                )

                self.console.print(table)
            else:
                self.console.print("[yellow]No valid prediction scores found[/yellow]")

    def _display_recommendations(self, project_stats, ml_backends, predictions):
        """Display recommendations for optimizing active learning."""

        recommendations = []

        # Check annotation progress
        completion_rate = (
            project_stats.get("completion_rate", 0) if project_stats else 0
        )
        if completion_rate < 20:
            recommendations.append(
                "🚀 Consider annotating more tasks to improve model performance"
            )
        elif completion_rate > 80:
            recommendations.append(
                "🎉 High completion rate - model should be performing well"
            )

        # Check ML backend status
        if not ml_backends:
            recommendations.append(
                "⚠️ No ML backends configured - add one to enable predictions"
            )

        # Check prediction availability and quality
        if not predictions:
            recommendations.append(
                "🔄 No predictions available - ensure ML backend is running"
            )
        else:
            # Analyze prediction confidence distribution
            scores = []
            for pred in predictions:
                if isinstance(pred, dict):
                    score = pred.get("score")
                    if score is None and pred.get("result"):
                        for result_item in pred.get("result", []):
                            if isinstance(result_item, dict) and "score" in result_item:
                                score = result_item["score"]
                                break
                    if score is not None:
                        scores.append(score)

            if scores:
                avg_confidence = sum(scores) / len(scores)
                low_confidence_count = sum(1 for s in scores if s < 0.5)

                if avg_confidence < 0.6:
                    recommendations.append(
                        "🎯 Many uncertain predictions - excellent for active learning!"
                    )
                elif avg_confidence > 0.9:
                    recommendations.append(
                        "✅ Model very confident - consider harder examples or new domains"
                    )

                if low_confidence_count > 10:
                    recommendations.append(
                        f"🔥 {low_confidence_count} high-value tasks ready for annotation"
                    )

        # Check for training needs
        annotated_tasks = (
            project_stats.get("annotated_tasks", 0) if project_stats else 0
        )
        if annotated_tasks > 0 and annotated_tasks % 10 == 0:
            recommendations.append("🔄 Consider retraining model with new annotations")

        # Active learning specific recommendations
        if predictions and len(predictions) > 50:
            recommendations.append(
                "📊 Sufficient predictions for effective active learning batch selection"
            )
        elif predictions and len(predictions) < 10:
            recommendations.append(
                "⚠️ Few predictions available - may need more diverse initial training data"
            )

        if recommendations:
            self.console.print(
                Panel(
                    "\n".join(recommendations),
                    title="[bold blue]Active Learning Recommendations[/bold blue]",
                    title_align="left",
                )
            )
        else:
            self.console.print(
                Panel(
                    "✅ Active learning setup looks good! Ready to start intelligent sampling.",
                    title="[bold green]Status: Ready[/bold green]",
                    title_align="left",
                )
            )
