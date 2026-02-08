"""Test WalkieVectorDB: upsert scene + objects from WalkieVision, then run queries.

Run from project root:
    uv run python db_test.py

Uses a separate persist dir (chroma_db_test) so the main DB is untouched.
"""

from dotenv import load_dotenv

load_dotenv()

from src.db import WalkieVectorDB, ObjectRecord, SceneRecord
from src.vision import WalkieVision
import cv2


def main():
    persist_dir = "chroma_db_test"

    print("WalkieVectorDB test")
    print("=" * 60)

    vision = WalkieVision(
        caption_provider="paligemma",
        embedding_provider="clip",
        detection_provider="yolo",
    )

    try:
        img = cv2.imread("test.jpg")
        objects = vision.detect_objects(img)
        for obj in objects:
            print(obj)
    except Exception as e:
        print(f"Error: {e}")
        raise e


    print("\nDone. DB persisted under", persist_dir)


if __name__ == "__main__":
    main()
