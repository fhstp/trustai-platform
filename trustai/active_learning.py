# active_learning_manager.py
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

from .base import APIClient


class UncertaintySampling:
    """Implements various uncertainty sampling strategies for active learning."""

    def __init__(self, strategy="entropy"):
        """
        Initialize uncertainty sampling.

        Args:
            strategy: 'entropy', 'least_confidence', 'margin', or 'ratio'
        """
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)

    def least_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate uncertainty using least confidence sampling.

        Args:
            probabilities: Array of shape (n_samples, n_classes) containing prediction probabilities

        Returns:
            Array of uncertainty scores where higher values indicate more uncertainty
        """
        most_confident = np.max(probabilities, axis=1)
        num_classes = probabilities.shape[1]
        normalized_uncertainty = (1 - most_confident) * (
            num_classes / (num_classes - 1)
        )
        return normalized_uncertainty

    def margin_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate uncertainty using margin of confidence sampling."""
        sorted_probs = np.sort(probabilities, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        return 1 - margin

    def entropy_based(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate uncertainty using entropy-based sampling."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

        entropy = -np.sum(probabilities * np.log2(probabilities), axis=1)
        max_entropy = np.log2(probabilities.shape[1])
        normalized_entropy = entropy / max_entropy
        return normalized_entropy

    def calculate_uncertainty(self, predictions: List[Dict]) -> List[Tuple[int, float]]:
        """
        Calculate uncertainty scores for predictions.

        Args:
            predictions: List of prediction objects from Label Studio API

        Returns:
            List of (task_id, uncertainty_score) tuples sorted by uncertainty (highest first)
        """
        uncertainties = []

        for pred in predictions:
            task_id = pred.get("task")

            if not task_id:
                continue

            # Get prediction confidence from multiple possible sources
            prediction_score = pred.get("score")  # Main prediction score
            result = pred.get("result", [])

            if not prediction_score and result:
                # Try to get score from result items
                for result_item in result:
                    if isinstance(result_item, dict) and "score" in result_item:
                        prediction_score = result_item["score"]
                        break

            if prediction_score is None:
                continue  # Skip predictions without confidence scores

            uncertainty = self._calculate_single_uncertainty(result, prediction_score)
            uncertainties.append((task_id, uncertainty))

        # Sort by uncertainty (highest first)
        return sorted(uncertainties, key=lambda x: x[1], reverse=True)

    def _calculate_single_uncertainty(
        self, result: List[Dict], confidence: float
    ) -> float:
        """Calculate uncertainty for a single prediction."""

        if self.strategy == "least_confidence":
            return 1.0 - confidence

        elif self.strategy == "entropy":
            # For classification tasks, calculate entropy based on confidence
            try:
                if result and isinstance(result[0], dict):
                    # Check if we have choice-based results
                    result_item = result[0]
                    value = result_item.get("value", {})

                    if "choices" in value:
                        # For single choice, use confidence directly
                        prob = confidence
                        if prob <= 0 or prob >= 1:
                            return 0.0  # No uncertainty for extreme values

                        # Calculate binary entropy (assuming binary classification)
                        entropy = -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)
                        return entropy
                    else:
                        # For other annotation types, use confidence-based uncertainty
                        return 1.0 - confidence

            except (IndexError, KeyError, TypeError, ValueError):
                pass

            # Fallback to least confidence
            return 1.0 - confidence

        elif self.strategy == "margin":
            # Simplified margin calculation for single confidence score
            # Convert to margin-like uncertainty (distance from decision boundary)
            return 1.0 - abs(confidence - 0.5) * 2

        elif self.strategy == "random":
            return np.random.random()

        return 1.0 - confidence


class DiversitySampling:
    """Implements diversity sampling strategies for active learning."""

    def __init__(self, strategy="kmeans", n_clusters=None):
        """
        Initialize diversity sampling.

        Args:
            strategy: 'kmeans', 'coreset', or 'outlier'
            n_clusters: Number of clusters for k-means (defaults to batch_size)
        """
        self.strategy = strategy
        self.n_clusters = n_clusters
        self.logger = logging.getLogger(__name__)

    def kmeans_diversity(
        self,
        features: np.ndarray,
        batch_size: int,
        uncertainty_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Select diverse samples using k-means clustering.

        Args:
            features: Feature representations of unlabeled samples
            batch_size: Number of samples to select
            uncertainty_scores: Optional uncertainty scores to weight selection

        Returns:
            Indices of selected samples
        """
        n_clusters = self.n_clusters or min(batch_size, len(features))

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

        selected_indices = []

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            if uncertainty_scores is not None:
                # Select the most uncertain sample from this cluster
                cluster_uncertainties = uncertainty_scores[cluster_indices]
                best_in_cluster = cluster_indices[np.argmax(cluster_uncertainties)]
            else:
                # Select the sample closest to the cluster centroid
                cluster_features = features[cluster_indices]
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.sum((cluster_features - centroid) ** 2, axis=1)
                best_in_cluster = cluster_indices[np.argmin(distances)]

            selected_indices.append(best_in_cluster)

        # If we need more samples, add the most uncertain remaining ones
        while len(selected_indices) < batch_size and uncertainty_scores is not None:
            remaining_mask = np.ones(len(uncertainty_scores), dtype=bool)
            remaining_mask[selected_indices] = False
            remaining_indices = np.where(remaining_mask)[0]

            if len(remaining_indices) == 0:
                break

            remaining_uncertainties = uncertainty_scores[remaining_indices]
            best_remaining = remaining_indices[np.argmax(remaining_uncertainties)]
            selected_indices.append(best_remaining)

        return np.array(selected_indices[:batch_size])

    def select_diverse_samples(
        self,
        features: np.ndarray,
        batch_size: int,
        uncertainty_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Select diverse samples using the configured strategy.

        Args:
            features: Feature representations
            batch_size: Number of samples to select
            uncertainty_scores: Optional uncertainty scores

        Returns:
            Indices of selected samples
        """
        if self.strategy == "kmeans":
            return self.kmeans_diversity(features, batch_size, uncertainty_scores)
        else:
            raise ValueError(f"Unknown diversity strategy: {self.strategy}")


class ActiveLearningManager:
    """Main class for managing the active learning workflow."""

    def __init__(
        self,
        api_client: APIClient,
        project_id: int,
        uncertainty_strategy: str = "entropy",
        diversity_strategy: str = "kmeans",
        batch_size: int = 10,
        uncertainty_weight: float = 0.7,
        diversity_weight: float = 0.3,
        min_confidence_threshold: float = 0.3,
    ):
        """
        Initialize the Active Learning Manager.

        Args:
            api_client: Label Studio API client instance
            project_id: ID of the Label Studio project
            uncertainty_strategy: Strategy for uncertainty sampling
            diversity_strategy: Strategy for diversity sampling
            batch_size: Number of samples to select per iteration
            uncertainty_weight: Weight for uncertainty component (0-1)
            diversity_weight: Weight for diversity component (0-1)
            min_confidence_threshold: Minimum confidence threshold for selecting tasks
        """
        self.api_client = api_client
        self.project_id = project_id
        self.batch_size = batch_size
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.min_confidence_threshold = min_confidence_threshold

        # Initialize sampling strategies
        self.uncertainty_sampler = UncertaintySampling(uncertainty_strategy)
        self.diversity_sampler = DiversitySampling(diversity_strategy)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Track state
        self.labeled_task_ids = set()
        self.current_iteration = 0

    def get_ml_predictions(self, task_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions from the connected ML backend for given tasks.

        Args:
            task_ids: List of task IDs to get predictions for

        Returns:
            Tuple of (probabilities, features) where:
            - probabilities: Shape (n_tasks, n_classes) prediction probabilities
            - features: Shape (n_tasks, n_features) feature representations
        """
        predictions = []
        features = []

        for task_id in task_ids:
            # Get predictions from ML backend via Label Studio API
            task_predictions = self.api_client.list_predictions(task_id=task_id)

            if task_predictions and len(task_predictions) > 0:
                # Extract probability distribution and features from prediction
                prediction = task_predictions[0]  # Get most recent prediction

                # Extract probabilities (this depends on your ML backend format)
                probs = self._extract_probabilities(prediction)
                feat = self._extract_features(prediction)

                predictions.append(probs)
                features.append(feat)
            else:
                self.logger.warning(f"No predictions found for task {task_id}")

        return np.array(predictions), np.array(features)

    def _extract_probabilities(self, prediction: Dict) -> np.ndarray:
        """Extract probability distribution from ML backend prediction."""
        # This implementation depends on your ML backend format
        # Example for classification with score field:
        if "score" in prediction:
            # Assuming binary classification for simplicity
            score = prediction["score"]
            return np.array([1 - score, score])
        else:
            # Default to uniform distribution if no score
            return np.array([0.5, 0.5])

    def _extract_features(self, prediction: Dict) -> np.ndarray:
        """Extract feature representation from ML backend prediction."""
        # This needs to be implemented based on your ML backend
        # For now, return dummy features
        return np.random.random(128)  # Replace with actual feature extraction

    def select_next_batch(self) -> List[int]:
        """
        Select the next batch of tasks for annotation using active learning.

        Returns:
            List of task IDs to annotate
        """
        self.logger.info(
            f"Starting batch selection for iteration {self.current_iteration + 1}"
        )

        # Get all predictions for the project
        predictions = self.api_client.list_predictions(project_id=self.project_id)
        if not predictions:
            self.logger.warning("No predictions available. Training model first...")
            return []

        # Get all tasks
        tasks_response = self.api_client.get_tasks(self.project_id)
        all_tasks = tasks_response.get("tasks", [])

        # Filter out already annotated tasks
        unannotated_tasks = [
            task
            for task in all_tasks
            if not task.get("annotations") and task["id"] not in self.labeled_task_ids
        ]

        if not unannotated_tasks:
            self.logger.info("No unannotated tasks available")
            return []

        # Filter predictions for unannotated tasks
        task_ids_set = {task["id"] for task in unannotated_tasks}
        relevant_predictions = [
            pred
            for pred in predictions
            if pred.get("task") in task_ids_set and pred.get("score") is not None
        ]

        if not relevant_predictions:
            self.logger.warning(
                "No predictions with valid scores for unannotated tasks"
            )
            return []

        # Calculate uncertainties
        uncertainties = self.uncertainty_sampler.calculate_uncertainty(
            relevant_predictions
        )
        # print(uncertainties)

        # Filter by confidence threshold
        high_uncertainty_tasks = [
            (task_id, uncertainty)
            for task_id, uncertainty in uncertainties
            if uncertainty >= self.min_confidence_threshold
        ]

        if not high_uncertainty_tasks:
            self.logger.info("No tasks above uncertainty threshold")
            # If no high uncertainty tasks, take the most uncertain ones available
            if uncertainties:
                batch_size = min(self.batch_size, len(uncertainties))
                high_uncertainty_tasks = uncertainties[:batch_size]
            else:
                return []

        # For now, just select the top uncertain tasks without diversity sampling
        # TODO: Implement proper feature extraction for diversity sampling
        high_uncertainty_tasks = sorted(high_uncertainty_tasks, key=lambda x: -x[1])
        # print(high_uncertainty_tasks)
        selected_task_ids = [
            task_id for task_id, _ in high_uncertainty_tasks[: self.batch_size]
        ]

        self.current_iteration += 1
        self.logger.info(f"Selected {len(selected_task_ids)} tasks for annotation")

        # Log detailed information about selected tasks
        for task_id in selected_task_ids:
            uncertainty = next(
                (u for tid, u in high_uncertainty_tasks if tid == task_id), None
            )
            self.logger.info(
                f"Selected task {task_id} with uncertainty score: {uncertainty:.3f}"
            )

        return selected_task_ids

    def trigger_retraining(self, project_id: int, backend_id: int) -> bool:
        """Update the set of labeled task IDs."""
        data = {
            "fields": "all",
            "project": project_id,
            "page_size": 1000,
            "only_annotated": True,
        }

        response = self.api_client._make_request("GET", f"/api/tasks/", params=data)
        tasks = [response.json()]
        self.logger.info(
            f"Retrieved {len(self.labeled_task_ids)} tasks for retraining."
        )
        backend_info = self.api_client.get_ml_backend(backend_id)
        if backend_info is None:
            self.logger.error(f"No backend with id {backend_id} found")
            return False
        project_info = self.api_client.get_project(project_id)
        if project_info is None:
            self.logger.error(f"No project with id {backend_id} found")
            return False
        request = requests.post(
            f"{backend_info['url'].rstrip('/')}/retrain",
            json={
                "project_id": project_id,
                "label_config": project_info["label_config"],
                "tasks": tasks,
            },
        )
        return request.json()["success"]

    def trigger_model_retraining(self) -> bool:
        """
        Trigger model retraining with newly annotated data.

        Returns:
            True if training was triggered successfully
        """
        # Get ML backends for the project
        ml_backends = self.api_client.list_ml_backends(self.project_id)

        if not ml_backends:
            self.logger.error("No ML backends available for training")
            return False

        # Train all available backends
        success = True
        for backend in ml_backends:
            backend_id = backend["id"]
            self.logger.info(f"Triggering training for backend {backend_id}")

            result = self.api_client.train_ml_backend(backend_id)
            if not result:
                self.logger.error(f"Failed to train backend {backend_id}")
                success = False

        return success

    def wait_for_training_completion(self, timeout: int = 300) -> bool:
        """
        Wait for model training to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if training completed successfully
        """
        ml_backends = self.api_client.list_ml_backends(self.project_id)
        if not ml_backends:
            return False

        start_time = time.time()

        while time.time() - start_time < timeout:
            all_ready = True

            for backend in ml_backends:
                status = self.api_client.get_training_status(backend["id"])
                if not status or status.get("readable_state") == "Training":
                    all_ready = False
                    break

            if all_ready:
                self.logger.info("All models finished training")
                return True

            time.sleep(10)  # Wait 10 seconds before checking again

        self.logger.warning(f"Training timeout after {timeout} seconds")
        return False

    def get_learning_stats(self) -> Dict:
        """Get statistics about the active learning process."""
        return {
            "current_iteration": self.current_iteration,
            "labeled_tasks": len(self.labeled_task_ids),
            "batch_size": self.batch_size,
            "uncertainty_strategy": self.uncertainty_sampler.strategy,
            "diversity_strategy": self.diversity_sampler.strategy,
            "min_confidence_threshold": self.min_confidence_threshold,
            "timestamp": datetime.now().isoformat(),
        }


class WebhookHandler:
    """Handles webhook events from Label Studio for automation."""

    def __init__(
        self,
        active_learning_manager: ActiveLearningManager,
        retraining_threshold: int = 10,
    ):
        """
        Initialize webhook handler.

        Args:
            active_learning_manager: The active learning manager instance
            retraining_threshold: Number of new annotations before triggering retraining
        """
        self.al_manager = active_learning_manager
        self.retraining_threshold = retraining_threshold
        self.new_annotations_count = 0
        self.logger = logging.getLogger(__name__)

    def handle_annotation_created(self, payload: Dict):
        """Handle annotation created webhook event."""
        task_id = payload.get("task", {}).get("id")
        if task_id:
            self.al_manager.trigger_retraining()([task_id])
            self.new_annotations_count += 1

            if self.new_annotations_count >= self.retraining_threshold:
                self.logger.info(
                    f"Reached retraining threshold ({self.retraining_threshold} annotations)"
                )
                self.al_manager.trigger_model_retraining()
                self.new_annotations_count = 0

    def handle_annotation_updated(self, payload: Dict):
        """Handle annotation updated webhook event."""
        # Could implement logic for handling annotation updates
        pass

    def process_webhook(self, event_type: str, payload: Dict):
        """Process incoming webhook events."""
        if event_type == "ANNOTATION_CREATED":
            self.handle_annotation_created(payload)
        elif event_type == "ANNOTATION_UPDATED":
            self.handle_annotation_updated(payload)
        else:
            self.logger.debug(f"Unhandled webhook event: {event_type}")


# Example usage integration with existing base_helper.py
class EnhancedHelper:
    """Extended helper class with active learning capabilities."""

    def __init__(self, base_helper):
        """Initialize with existing base helper."""
        self.base_helper = base_helper
        self.al_manager = None
        self.webhook_handler = None

    def setup_active_learning(self, project_id: int, **kwargs):
        """Setup active learning for a project."""
        if not self.base_helper.client:
            print("Error: API client not initialized")
            return False

        self.al_manager = ActiveLearningManager(
            api_client=self.base_helper.client, project_id=project_id, **kwargs
        )

        self.webhook_handler = WebhookHandler(self.al_manager)
        print(f"Active learning setup complete for project {project_id}")
        return True

    def run_active_learning_iteration(self):
        """Run one iteration of active learning."""
        if not self.al_manager:
            print("Error: Active learning not setup")
            return False

        # Select next batch
        selected_tasks = self.al_manager.select_next_batch()

        if not selected_tasks:
            print("No more tasks to annotate")
            return False

        print(f"Selected {len(selected_tasks)} tasks for annotation: {selected_tasks}")

        # Here you would typically launch the annotation interface
        # or programmatically create the annotation batch

        return True

    def start_active_learning_workflow(self):
        """Start the complete active learning workflow."""
        if not self.al_manager:
            print("Error: Active learning not setup")
            return

        print("Starting active learning workflow...")

        iteration = 0
        max_iterations = 50  # Safety limit

        while iteration < max_iterations:
            print(f"\n--- Active Learning Iteration {iteration + 1} ---")

            if not self.run_active_learning_iteration():
                print("Active learning workflow completed")
                break

            # In a real implementation, you would wait for annotations
            # and trigger retraining via webhooks

            iteration += 1

        print("Active learning workflow finished")
