import uuid
from dotenv import load_dotenv

load_dotenv()

from src.db import WalkieVectorDB, ObjectRecord, SceneRecord
from src.vision import WalkieVision


def main():
    print("Mocking data for WalkieVectorDB")
    print("=" * 60)

    vision = WalkieVision(
        caption_provider="paligemma",
        embedding_provider="clip",
        detection_provider="yolo",
        preload=True,
    )
    
    # 2. Upsert into DB
    db = WalkieVectorDB()
    db.delete_all()
    scene_id = f"scene_{uuid.uuid4().hex[:8]}"
    
    objects = [
        ObjectRecord(
            object_id=f"obj_{uuid.uuid4().hex[:8]}",
            object_xyz=[5.189, -0.215, 0.0],
            object_embedding=vision.embed_text("table"),
            heading=0.0293959,
            scene_id=scene_id,
            class_id=1,
            class_name="Table",
        ),
        ObjectRecord(
            object_id=f"obj_{uuid.uuid4().hex[:8]}",
            object_xyz=[3.275, -2.783, 0.0],
            object_embedding=vision.embed_text("cabinet"),
            heading=-1.5352134,
            scene_id=scene_id,
            class_id=2,
            class_name="White Cabinet",
        ),
        ObjectRecord(
            object_id=f"obj_{uuid.uuid4().hex[:8]}",
            object_xyz=[6.312, 0.127, 0.0],
            object_embedding=vision.embed_text("chair"),
            heading=-1.5352134,
            scene_id=scene_id,
            class_id=3,
            class_name="Chair",
        ),
        ObjectRecord(
            object_id=f"obj_{uuid.uuid4().hex[:8]}",
            object_xyz=[6.153, -0.322, 0.0],
            object_embedding=vision.embed_text("chair"),
            heading=-1.5352134,
            scene_id=scene_id,
            class_id=3,
            class_name="Chair",
        ),
        ObjectRecord(
            object_id=f"obj_{uuid.uuid4().hex[:8]}",
            object_xyz=[5.572, -0.775, 0.0],
            object_embedding=vision.embed_text("chair"),
            heading=-1.5352134,
            scene_id=scene_id,
            class_id=3,
            class_name="Chair",
        ),
        ObjectRecord(
            object_id=f"obj_{uuid.uuid4().hex[:8]}",
            object_xyz=[5.472, 0.515, 0.0],
            object_embedding=vision.embed_text("chair"),
            heading=-1.5352134,
            scene_id=scene_id,
            class_id=3,
            class_name="Chair",
        ),
        # ObjectRecord(
        #     object_id=f"obj_{uuid.uuid4().hex[:8]}",
        #     object_xyz=[0.0, 0.0, 0.0],
        #     object_embedding=vision.embed_text("chair"),
        #     heading=0.0,
        #     scene_id=scene_id,
        #     class_id=3,
        #     class_name="chair",
        # ),
    ]
    
    for obj in objects:
        db.upsert_object(obj)
    
    query_result = db.query_objects(vision.embed_text("water bottle"))
    print(query_result)


if __name__ == "__main__":
    main()