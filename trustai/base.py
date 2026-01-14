#!/usr/bin/env python3
"""API Client for Labelstudio"""

import os
import sys
from typing import Any, Dict, List, Optional

import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install with: pip install requests")
    sys.exit(1)


@dataclass
class Config:
    """Configuration dataclass for Label Studio connection."""

    api_base_url: str
    api_key: str
    timeout: int = 30
    verify_ssl: bool = True


class APIClient:
    """
    Professional Label Studio API Client with comprehensive functionality.
    """

    def __init__(self, config: Config):
        """
        Initialize the Label Studio API client.

        Args:
            config: Configuration object containing API details
        """
        self.config = config
        self.session = requests.Session()

        if not config.api_key:
            raise ValueError("API key is required")
        self.session.headers.update(
            {
                "Authorization": f"Token {config.api_key.strip()}",
                "Content-Type": "application/json",
            }
        )

        self.logger = self._setup_logger()

    def __del__(self):
        """Cleanup session on destruction."""
        if hasattr(self, "session"):
            self.session.close()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("labelstudio_client")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make HTTP request to Label Studio API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            **kwargs: Additional request parameters

        Returns:
            Response object

        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.config.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        kwargs.setdefault("timeout", self.config.timeout)
        kwargs.setdefault("verify", self.config.verify_ssl)

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self.logger.error(f"API request failed: {method} {url} - {e}")
            raise

    def test_connection(self) -> bool:
        """
        Test connection to Label Studio API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self._make_request("GET", "/api/projects/")
            self.logger.info("Connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    def get_projects(self) -> List[Dict[str, Any]]:
        """
        Get all projects from Label Studio.

        Returns:
            List of project dictionaries
        """
        try:
            response = self._make_request("GET", "/api/projects/")
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get projects: {e}")
            return []

    def create_project(
        self,
        title: str,
        description: str = "",
        label_config: str = "",
        sampling: str = "Sequential sampling",
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new project in Label Studio.

        Args:
            title: Project title
            description: Project description
            label_config: Label configuration XML
            sampling: Sampling method

        Returns:
            Created project dictionary or None if failed
        """
        try:
            data = {
                "title": title,
                "description": description,
                "label_config": label_config,
                "sampling": sampling,
            }
            response = self._make_request("POST", "/api/projects/", json=data)
            project = response.json()
            self.logger.info(f"Created project: {title} (ID: {project.get('id')})")
            return project
        except Exception as e:
            self.logger.error(f"Failed to create project: {e}")
            return None

    def get_project(self, project_id: int) -> Optional[Dict[str, Any]]:
        """
        Get specific project details.

        Args:
            project_id: Project ID

        Returns:
            Project dictionary or None if not found
        """
        try:
            response = self._make_request("GET", f"/api/projects/{project_id}/")
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get project {project_id}: {e}")
            return None

    def delete_project(self, project_id: int) -> bool:
        """
        Delete a project.

        Args:
            project_id: Project ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self._make_request("DELETE", f"/api/projects/{project_id}/")
            self.logger.info(f"Deleted project {project_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete project {project_id}: {e}")
            return False

    def get_tasks(self, project_id: int, fields: str = "all") -> List[Dict[str, Any]]:
        """
        Get all tasks for a project.

        Args:
            project_id: Project ID

        Returns:
            List of task dictionaries
        """
        data = {"fields": fields, "project": project_id}

        try:
            response = self._make_request("GET", f"/api/tasks/", params=data)
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get tasks for project {project_id}: {e}")
            return []

    def delete_tasks(self, task_id: int) -> bool:
        """
        Delete a tasks from a project.

        Args:
            task_id: Task ID

        Returns:
             True if successful, False otherwise
        """

        try:
            self._make_request("DELETE", f"/api/tasks/{task_id}/")
            self.logger.info(f"Deleted task with {task_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete task with id {task_id}: {e}")
            return False

    def delete_all_tasks(self, project_id: int) -> bool:
        """
        Delete all tasks from a project.

        Args:
            project_id: Project ID

        Returns:
             True if successful, False otherwise
        """

        try:
            self._make_request("DELETE", f"/api/projects/{project_id}/tasks/")
            self.logger.info(f"Deleted all tasks")
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to delete all task for current project with id {project_id}: {e}"
            )
            return False

    def create_task(
        self, project_id: int, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new task in a project.

        Args:
            project_id: Project ID
            data: Task data dictionary

        Returns:
            Created task dictionary or None if failed
        """
        try:
            response = self._make_request(
                "POST", f"/api/projects/{project_id}/tasks/", json=data
            )
            task = response.json()
            self.logger.info(f"Created task in project {project_id}")
            return task
        except Exception as e:
            self.logger.error(f"Failed to create task in project {project_id}: {e}")
            return None

    def import_tasks(
        self,
        project_id: int,
        tasks: List[Dict[str, Any]],
        commit: bool = True,
        return_task_ids: bool = False,
        preannotated_from_fields: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Import multiple tasks to a project.

        Args:
            project_id: Project ID
            tasks: List of task data dictionaries
            commit: Set to True to immediately commit tasks to the project
            return_task_ids: Set to True to return task IDs in the response
            preannotated_from_fields: List of fields to preannotate from task data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare the payload
            params = {"commit_to_project": commit, "return_task_ids": return_task_ids}

            if preannotated_from_fields:
                params["preannotated_from_fields"] = preannotated_from_fields

            # Send tasks as JSON in the request body
            response = self._make_request(
                "POST", f"/api/projects/{project_id}/import", json=tasks, params=params
            )

            result = response.json()
            self.logger.info(
                f"Imported {result.get('task_count', len(tasks))} tasks to project {project_id}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Failed to import tasks to project {project_id}: {e}")
            return None

    def import_tasks_from_file(
        self,
        project_id: int,
        file_path: str,
        commit: bool = True,
        return_task_ids: bool = False,
        preannotated_from_fields: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Import tasks from a file (JSON, CSV, TSV, TXT).

        Args:
            project_id: Project ID
            file_path: Path to the file containing tasks
            commit: Set to True to immediately commit tasks to the project
            return_task_ids: Set to True to return task IDs in the response
            preannotated_from_fields: List of fields to preannotate from task data

        Returns:
            Import response dictionary or None if failed
        """
        try:
            import os

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Prepare the payload
            data = {"commit_to_project": commit, "return_task_ids": return_task_ids}

            if preannotated_from_fields:
                data["preannotated_from_fields"] = preannotated_from_fields

            # Open and send file
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f)}

                # Remove Content-Type header for file uploads
                headers = self.session.headers.copy()
                if "Content-Type" in headers:
                    del headers["Content-Type"]

                response = self._make_request(
                    "POST",
                    f"/api/projects/{project_id}/import",
                    files=files,
                    params=data,
                    headers=headers,
                )

            result = response.json()
            self.logger.info(f"Imported tasks from {file_path} to project {project_id}")
            return result

        except Exception as e:
            self.logger.error(
                f"Failed to import tasks from file {file_path} to project {project_id}: {e}"
            )
            return None

    def refresh_project(self, project_id: int) -> Optional[Dict[str, Any]]:
        """
        Refresh project data from server.

        Args:
            project_id: Project ID

        Returns:
            Updated project dictionary or None if failed
        """
        try:
            response = self._make_request("GET", f"/api/projects/{project_id}/")
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to refresh project {project_id}: {e}")
            return None

    # def import_tasks_from_url(self, project_id: int, url: str,
    #                      commit: bool = True, return_task_ids: bool = False,
    #                      preannotated_from_fields: List[str] = None) -> Optional[Dict[str, Any]]:
    #     """
    #     Import tasks from a URL.

    #     Args:
    #         project_id: Project ID
    #         url: URL to the file containing tasks
    #         commit: Set to True to immediately commit tasks to the project
    #         return_task_ids: Set to True to return task IDs in the response
    #         preannotated_from_fields: List of fields to preannotate from task data

    #     Returns:
    #         Import response dictionary or None if failed
    #     """
    #     try:
    #         # Prepare the payload
    #         data = {
    #             "url": url,
    #             "commit_to_project": commit,
    #             "return_task_ids": return_task_ids
    #         }

    #         if preannotated_from_fields:
    #             data["preannotated_from_fields"] = preannotated_from_fields

    #         response = self._make_request(
    #             'POST',
    #             f'/api/projects/{project_id}/import',
    #             json=data
    #         )

    #         result = response.json()
    #         self.logger.info(f"Imported tasks from URL {url} to project {project_id}")
    #         return result

    #     except Exception as e:
    #         self.logger.error(f"Failed to import tasks from URL {url} to project {project_id}: {e}")
    #         return None

    def get_annotations(self, project_id: int, task_id: int) -> List[Dict[str, Any]]:
        """
        Get annotations for a specific task.

        Args:
            project_id: Project ID
            task_id: Task ID

        Returns:
            List of annotation dictionaries
        """
        try:
            response = self._make_request("GET", f"/api/tasks/{task_id}/annotations/")
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get annotations for task {task_id}: {e}")
            return []

    def export_annotations(
        self, project_id: int, export_format: str = "JSON"
    ) -> Optional[str]:
        """
        Export annotations from a project.

        Args:
            project_id: Project ID
            export_format: Export format (JSON, CSV, etc.)

        Returns:
            Export data as string or None if failed
        """
        try:
            params = {"exportType": export_format}
            response = self._make_request(
                "GET", f"/api/projects/{project_id}/export", params=params
            )
            self.logger.info(f"Exported annotations from project {project_id}")
            return response.text
        except Exception as e:
            self.logger.error(
                f"Failed to export annotations from project {project_id}: {e}"
            )
            return None

    def list_ml_backends(self, project_id: int) -> Optional[list]:
        """
        List all ML backends for a project.

        Args:
            project_id: Project ID

        Returns:
            List of ML backends or None if failed
        """
        try:
            params = {"project": {project_id}}
            response = self._make_request("GET", f"/api/ml", params=params)
            self.logger.info(f"Listed ML backends for project {project_id}")
            return response.json()
        except Exception as e:
            self.logger.error(
                f"Failed to list ML backends for project {project_id}: {e}"
            )
            return None

    def add_ml_backend(
        self,
        project_id: int,
        url: str,
        desc: str = "",
        title: str = None,
        is_interactive: bool = False,
        auth_method: str = "NONE",
        basic_auth_user: str = "string",
        basic_auth_pass: str = "string",
        extra_params: Dict = {},
        timeout: int = 0,
    ) -> Optional[dict]:
        """
        Add an ML backend to a project.

        Args:
            project_id: Project ID
            url: ML backend URL
            desc: Description (optional)
            title: Display name (optional)
            is_interactive: Enable interactive mode (optional)
            auth_method: Auth method - 'NONE', 'BASIC_AUTH', or 'API_KEY' (optional)
            basic_auth_user: Username for BASIC_AUTH (optional)
            basic_auth_pass: Password for BASIC_AUTH (optional)
            extra_params: Extra config params (optional)
            timeout: Request timeout in seconds (optional)

        Returns:
            ML backend info as dict if successful, None otherwise
        """

        try:
            data = {
                "url": url,
                "project": project_id,
                "is_interactive": is_interactive,
                "title": title,
                "description": desc,
                "auth_method": "NONE",
                "basic_auth_user": "string",
                "basic_auth_pass": "string",
                "extra_params": {},
                "timeout": 0,
            }

            response = self._make_request("POST", "/api/ml/", json=data)
            project = response.json()
            self.logger.info(f"ML Backend added to {project.get('id')})")
            return project
        except Exception as e:
            self.logger.error(f"Failed to add ML backend: {e}")
            return None

    def remove_ml_backend(self, project_id: int, backend_id: int) -> bool:
        """
        Remove an ML backend from a project.

        Args:
            project_id: Project ID
            backend_id: ML backend ID

        Returns:
            True if successful, False otherwise
        """
        try:
            self._make_request("DELETE", f"/api/ml/{backend_id}")
            self.logger.info(
                f"Removed ML backend {backend_id} from project {project_id}"
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to remove ML backend {backend_id} from project {project_id}: {e}"
            )
            return False

    def get_ml_backend(self, backend_id: int) -> Optional[dict]:
        """
        Retrieves info for backend

        Args:
            backend_id: Backend ID

        Returns:
            Info for backend if successful, None otherwise
        """
        try:
            response = self._make_request("GET", f"/api/ml/{backend_id}")
            self.logger.info(f"Retrieved info for backend {backend_id}")
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to retrieve info for backend {backend_id}: {e}")
            return None

    def create_import_storage(
        self,
        project_id: int,
        title: str,
        desc: str,
        path: str,
        regex_filter: str,
        use_blob_urls: bool,
    ) -> Optional[dict]:
        data = {
            "title": title,
            "description": desc,
            "project": project_id,
            "path": path,
            "regex_filter": regex_filter,
            "use_blob_urls": use_blob_urls,
        }

        try:

            response = self._make_request(
                "POST", f"/api/storages/localfiles/", json=data
            )
            if response:
                self.logger.info(f"Local Storage added to project with id {project_id}")
            return response.json()
        except Exception as e:
            self.logger.error(
                f"[red]Failed to create local import storage {title} for project {project_id}: {e}[/red]"
            )
            return False

    def sync_import_storage(self, storage_id: int) -> bool:
        """_summary_

        Args:
            storage_id (int): _description_

        Returns:
            bool: _description_
        """
        data = {"id": storage_id}

        try:
            response = self._make_request(
                "POST", f"/api/storages/localfiles/{storage_id}/sync", json=data
            )
            self.logger.info(f"Import for local storage {storage_id} synced")
            return response.json()
        except Exception as e:
            self.logger.error(
                f"Failed to sync import for local storage {storage_id}: {e}"
            )
            return False

    def get_all_import_storage(self, project_id: int) -> Optional[dict]:
        """Get all local import storages for project

        Args:
            project_id (int): Project ID

        Returns:
            Dict: returns all local storages for current project
        """

        try:
            params = {"project": project_id}
            response = self._make_request(
                "GET", "/api/storages/localfiles", params=params
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to local storage: {e}")
            return []

    def remove_import_storage(self, project_id: int, storage_id: int) -> bool:
        """
        Remove local storage from a project.

        Args:
            project_id: Project ID
            storage_id: Local Storage ID

        Returns:
            True if successful, False otherwise
        """
        try:
            self._make_request("DELETE", f"/api/storages/localfiles/{storage_id}")
            self.logger.info(
                f"Removed local storage {storage_id} from project {project_id}"
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to remove local storage {storage_id} from project {project_id}: {e}"
            )
            return False

    def train_ml_backend(self, backend_id: int) -> Optional[Dict[str, Any]]:
        """
        Start training for a specific ML backend.

        Args:
            backend_id: ML backend ID

        Returns:
            Training response dictionary or None if failed
        """
        data = {"use_ground_truth": True}

        try:
            response = self._make_request(
                "POST", f"/api/ml/{backend_id}/train", json=data
            )
            # Check if response has content before trying to parse JSON
            if response.content:
                try:
                    result = response.json()
                except ValueError:
                    # If JSON parsing fails, create a basic success response
                    result = {
                        "status": "training_started",
                        "message": "Training initiated successfully",
                    }
            else:
                # Empty response typically means success for training endpoint
                result = {
                    "status": "training_started",
                    "message": "Training initiated successfully",
                }

            self.logger.info(f"Started training for ML backend {backend_id}")
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to start training for ML backend {backend_id}: {e}"
            )
            return None

    def get_training_status(self, backend_id: int) -> Optional[Dict[str, Any]]:
        """
        Get training status for a specific ML backend.

        Args:
            backend_id: ML backend ID

        Returns:
            Training status dictionary or None if failed
        """
        try:
            response = self._make_request("GET", f"/api/ml/{backend_id}")
            # Check if response has content
            if response.content:
                result = response.json()
            else:
                result = {
                    "status": "unknown",
                    "message": "No status information available",
                }

            return result
        except Exception as e:
            self.logger.error(
                f"Failed to get training status for ML backend {backend_id}: {e}"
            )
            return None

    def list_predictions(
        self, task_id: int = None, project_id: int = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get a list of all predictions, optionally filtered by task or project.

        Args:
            task_id: Filter predictions by task ID (optional)
            project_id: Filter predictions by project ID (optional)

        Returns:
            List of prediction dictionaries or None if failed
        """
        try:
            params = {}
            if task_id:
                params["task"] = task_id
            if project_id:
                params["project"] = project_id

            response = self._make_request("GET", "/api/predictions/", params=params)
            result = response.json()
            self.logger.info(f"Retrieved {len(result)} predictions")
            return result

        except Exception as e:
            self.logger.error(f"Failed to list predictions: {e}")
            return None

    def get_prediction_details(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """
        Get details about a specific prediction by its ID.

        Args:
            prediction_id: Prediction ID

        Returns:
            Prediction details dictionary or None if failed
        """
        try:
            response = self._make_request("GET", f"/api/predictions/{prediction_id}/")
            result = response.json()
            self.logger.info(f"Retrieved prediction details for ID {prediction_id}")
            return result

        except Exception as e:
            self.logger.error(
                f"Failed to get prediction details for ID {prediction_id}: {e}"
            )
            return None

    def create_prediction(
        self,
        task_id: int,
        result: Dict[str, Any],
        score: float = None,
        model_version: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a prediction for a specific task.

        Args:
            task_id: Task ID for which the prediction is created
            result: Prediction result in JSON format
            score: Prediction score (optional)
            model_version: Model version tag (optional)

        Returns:
            Created prediction dictionary or None if failed
        """
        try:
            data = {"task": task_id, "result": result}

            if score is not None:
                data["score"] = score
            if model_version:
                data["model_version"] = model_version

            response = self._make_request("POST", "/api/predictions/", json=data)
            prediction = response.json()
            self.logger.info(f"Created prediction for task {task_id}")
            return prediction

        except Exception as e:
            self.logger.error(f"Failed to create prediction for task {task_id}: {e}")
            return None

    def delete_prediction(self, prediction_id: int) -> bool:
        """
        Delete a specific prediction.

        Args:
            prediction_id: Prediction ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self._make_request("DELETE", f"/api/predictions/{prediction_id}/")
            self.logger.info(f"Deleted prediction {prediction_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete prediction {prediction_id}: {e}")
            return False

    def create_annotation(
        self, task_id: int, result: List[Dict], completed_by: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Create an annotation for a task.

        Args:
            task_id: Task ID
            result: Annotation result
            completed_by: User ID who completed the annotation

        Returns:
            Created annotation dictionary or None if failed
        """
        try:
            data = {"task": task_id, "result": result, "completed_by": completed_by}

            response = self._make_request(
                "POST", f"/api/tasks/{task_id}/annotations/", json=data
            )
            annotation = response.json()
            self.logger.info(f"Created annotation for task {task_id}")
            return annotation
        except Exception as e:
            self.logger.error(f"Failed to create annotation for task {task_id}: {e}")
            return None

    def get_project_statistics(self, project_id: int) -> Optional[Dict[str, Any]]:
        """
        Get project statistics including task counts and completion rates.

        Args:
            project_id: Project ID

        Returns:
            Statistics dictionary or None if failed
        """
        try:
            tasks = self.get_tasks(project_id)
            if not tasks:
                return {"total_tasks": 0, "annotated_tasks": 0, "completion_rate": 0}

            total_tasks = len(tasks)
            annotated_tasks = 0

            for task in tasks["tasks"]:
                annotations = self.get_annotations(project_id, task["id"])
                if annotations:
                    annotated_tasks += 1

            completion_rate = (
                (annotated_tasks / total_tasks) * 100 if total_tasks > 0 else 0
            )

            return {
                "total_tasks": total_tasks,
                "annotated_tasks": annotated_tasks,
                "remaining_tasks": total_tasks - annotated_tasks,
                "completion_rate": completion_rate,
            }
        except Exception as e:
            self.logger.error(f"Failed to get project statistics for {project_id}: {e}")
            return None

    def create_webhook(
        self, project_id: int, url: str, events: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Create a webhook for the project."""
        try:
            data = {
                "url": url,
                "project": project_id,
                "events": events,
                "send_payload": True,
                "is_active": True,
                "send_for_all_actions": False,
                "actions": [
                    "ANNOTATION_CREATED",
                    "ANNOTATIONS_CREATED",
                    "ANNOTATION_UPDATED",
                    "ANNOTATIONS_DELETED",
                ],
            }
            response = self._make_request("POST", f"/api/webhooks/", json=data)
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to create webhook: {e}")
            return None

    def list_webhooks(self, project_id: int) -> List[Dict[str, Any]]:
        """List all webhooks for a project."""
        try:
            response = self._make_request(
                "GET", f"/api/projects/{project_id}/webhooks/"
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to list webhooks: {e}")
            return []

    def update_project_sampling(self, project_id: int, sampling_method: str) -> bool:
        """Update project sampling method for active learning."""
        try:
            data = {"sampling": sampling_method}
            response = self._make_request(
                "PATCH", f"/api/projects/{project_id}/", json=data
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to update sampling method: {e}")
            return False
