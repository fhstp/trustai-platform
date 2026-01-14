#!/usr/bin/env python3
"""
Example Active Learning Script for Label Studio

This example demonstrates how to:
1. Setup a project for active learning
2. Add sample data and ML backend
3. Run active learning iterations
4. Monitor progress and effectiveness

Usage:
    python -m trustai.run_active_learning
"""

from rich.prompt import Prompt

from .active_learning_runner import ActiveLearningRunner
from .base_helper import baseHelper


def main():
    """Demonstrate active learning workflow."""

    # Initialize helper
    helper = baseHelper()

    print("🤖 Active Learning Demo for Label Studio")
    print("=" * 50)

    # Load configuration
    if not helper.load_config():
        print("Failed to load configuration. Exiting.")
        return

    # Test connection
    helper.test_connection()

    # Create or select project
    print("\n1. Project Setup")
    projects = helper.client.get_projects()

    if not projects or not projects.get("results"):
        print("No projects found. Creating a new one...")
        helper.create_project_interactive()
    else:
        print("Select existing project or create new:")
        print("1. Select existing project")
        print("2. Create new project")

        choice = int(Prompt.ask("Choose option", choices=["1", "2"], default="1"))
        if choice == "2":
            helper.create_project_interactive()
        else:
            helper.select_project()

    if not helper.current_project:
        print("No project selected. Exiting.")
        return

    # Check if project has tasks
    print("\n2. Data Setup")
    tasks = helper.client.get_tasks(helper.current_project["id"])
    if not tasks or not tasks.get("tasks"):
        print("No tasks found. Adding sample data...")
        helper.import_sample_tasks()

    # Check if project has ML backend
    print("\n3. ML Backend Setup")
    backends = helper.client.list_ml_backends(helper.current_project["id"])
    if not backends:
        print("No ML backend found. Please add one...")
        helper.add_ml_backend()

        # Wait for backend to be ready
        print("Waiting for ML backend to initialize...")
        import time

        time.sleep(10)

    # Start active learning session
    print("\n4. Active Learning Session")
    print("Starting active learning with uncertainty sampling...")
    helper.start_active_learning_session()

    print("\n🎉 Active Learning Demo Complete!")
    print("You can now continue with more iterations or analyze results.")


def run_automated_demo():
    """Run a fully automated active learning demo."""

    runner = ActiveLearningRunner()

    print("🚀 Starting Automated Active Learning Runner")
    print("=" * 50)

    runner.run()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--automated":
        run_automated_demo()
    else:
        main()
