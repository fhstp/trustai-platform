import io
import os
import base64
import requests
import numpy as np
from datetime import datetime
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase

# Hugging Face
from transformers import pipeline

# LIME
from lime import lime_image
from skimage.color import label2rgb


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def load_image_from_url(url: str) -> Image.Image:
    """Download an image from a URL and return a PIL RGB image."""
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class DogCatModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Parse Label Studio config
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema["to_name"][0]
        self.labels = schema["labels"]  # ["dog", "cat"]

        # Initialize Hugging Face classifier
        self.classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",
            device="cuda"
        )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _predict_probs(self, pil_img: Image.Image):
        """Return [dog_prob, cat_prob]."""
        preds = self.classifier(pil_img, top_k=None)
        if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], list):
            preds = preds[0]

        label_to_score = {p["label"].lower(): float(p["score"]) for p in preds}
        dog_score = sum(v for k, v in label_to_score.items() if "dog" in k)
        cat_score = sum(v for k, v in label_to_score.items() if "cat" in k)

        if dog_score == 0 and cat_score == 0:
            return np.array([0.5, 0.5])
        total = dog_score + cat_score
        return np.array([dog_score / total, cat_score / total])

    def _lime_explanation(self, pil_img: Image.Image):
        """Generate LIME explanation image showing both positive and negative regions."""
        np_img = np.array(pil_img)

        def classifier_fn(imgs):
            probs = []
            for im in imgs:
                pil = Image.fromarray(im.astype(np.uint8))
                probs.append(self._predict_probs(pil))
            return np.array(probs)

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np_img,
            classifier_fn,
            labels=(0, 1),
            num_samples=200,
        )

        probs = classifier_fn([np_img])[0]
        top_label = int(np.argmax(probs))

        # Both positive & negative regions
        temp, mask = explanation.get_image_and_mask(
            top_label,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )

        # Color overlay: green = positive, red = negative
        img_overlay = label2rgb(mask, temp, bg_label=0, alpha=0.4)
        img_uint8 = (img_overlay * 255).astype(np.uint8)
        lime_img = Image.fromarray(img_uint8)
        return lime_img

    # -----------------------------------------------------------------
    # Label Studio API methods
    # -----------------------------------------------------------------
    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            image_url = task["data"].get("image")
            pil_img = load_image_from_url(image_url)

            # Classification
            probs = self._predict_probs(pil_img)
            pred_idx = int(np.argmax(probs))
            pred_label = self.labels[pred_idx]
            score = float(probs[pred_idx])

            # LIME Explanation
            lime_img = self._lime_explanation(pil_img)
            explanation_path = task["data"].get("explanation")
            lime_img.save(explanation_path[explanation_path.index("PetImages"):])

            predictions.append(
                {
                    "task": task["id"],
                    "score": score,
                    "model_version": "dogcat-lime",
                    "result": [
                        {
                            "from_name": self.from_name,
                            "to_name": self.to_name,
                            "type": "choices",
                            "value": {"choices": [pred_label]},
                        },
                    ],
                }
            )
        print(predictions)

        return predictions

    def fit(self, annotations, **kwargs):
        return {"status": "not implemented"}

