import json

from pydantic import BaseModel


def detect_text(
    gcs_preview_image_path: str,
) -> tuple["vision.ImageAnnotatorClient", list[dict]]:
    """
    Initializes the Google Vision API client and detects text in an image file.

    Args:
        gcs_preview_image_path (str): in the format of 'images/date/filename.jpg'
    """
    from google.cloud import vision

    vision_client = vision.ImageAnnotatorClient()
    """Detects text in an image file using Google Vision API and returns (raw_response, results_list)."""
    uri = f"gs://recyclo-c0fd1.firebasestorage.app/{gcs_preview_image_path}"
    image = vision.Image(source=vision.ImageSource(gcs_image_uri=uri))
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations

    results = []
    for text in texts:
        # Each text has a bounding_poly with vertices
        box = [(v.x, v.y) for v in text.bounding_poly.vertices]
        results.append({"description": text.description, "bounding_poly": box})
    return response, results


class ShoppingListItem(BaseModel):
    item: str
    confidence: float


def make_gemini_client() -> "genai.Client":
    from google import genai
    from google.genai.types import HttpOptions

    return genai.Client(
        vertexai=True,
        project="recyclo-c0fd1",
        location="us-central1",
        http_options=HttpOptions(api_version="v1"),
    )


def gemini_item_from_text(text: str) -> tuple[str, float]:
    """
    Uses Gemini to generate a grocery list item and a confidence bool from OCR text.
    Returns (item: str, confidence: float)
    """
    gemini_client = make_gemini_client()
    prompt = (
        "Given the following partial text from a recycled item, "
        "Identify the most likely grocery list item. "
        "Produce a confidence value for the item label between 0 and 100. "
        "If it doesn't seem like an item purchaseable at a super market, return an empty string and low confidence. "
        f"Text: {text}"
    )
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": ShoppingListItem,
        },
    )

    result = json.loads(response.text)
    item = result.get("item", "")
    confident = result.get("confidence", 0)
    return item, confident


def gemini_grocery_list(items: list[str]) -> str:
    """
    Uses Gemini to generate a grocery list item and a confidence bool from OCR text.
    """
    text = "\n".join(items)
    gemini_client = make_gemini_client()

    prompt = (
        "Given the following partial items create a grocery list with one item per line. "
        "Do not include any additional text or formatting. "
        "Group similar items together."
        f"Text: {text}"
    )
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
    )

    return response.text


def image_to_item(gcs_image_path: str) -> str | None:
    """
    Uses OCR to extract text from an image and then uses Gemini to identify the most likely grocery list item.
    """
    _, results = detect_text(gcs_image_path)
    if not results:
        return None

    # Take the first result as the most relevant
    item_text = results[0]["description"]
    item, confidence = gemini_item_from_text(item_text)
    return item if confidence > 50 else None
