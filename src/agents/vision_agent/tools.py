"""Vision agent tools backed by WalkieVision and WalkieVectorDB."""

from __future__ import annotations

import uuid

from langchain_core.tools import tool

from src.db.walkie_db import ObjectRecord, SceneRecord, WalkieVectorDB
from src.vision import WalkieVision


def get_vision_tools(
    vision: WalkieVision,
    db: WalkieVectorDB | None = None,
) -> list:
    """Build vision tools that use WalkieVision and optionally WalkieVectorDB.

    Args:
        vision: WalkieVision instance (camera must be open when tools run).
        db: Optional WalkieVectorDB for find_object, find_scene, scan_and_remember.

    Returns:
        List of LangChain tools.
    """

    robot = vision._camera._bot

    # -------------------------------------------------------------------------
    # Scene description and classification
    # -------------------------------------------------------------------------

    # @tool
    # def describe_surroundings_from_view() -> str:
    #     """Get a general description of what the robot currently sees.

    #     Use this when the user asks what you see, describe the room, or look around.

    #     Returns:
    #         str: A description of the current scene and surroundings.
    #     """
    #     print(f"Describing surroundings")
    #     image = vision.capture()
    #     return vision.caption(image)

    @tool
    def classify_scene_from_view(categories: str) -> str:
        """Classify the current view into one of the given categories (e.g. room types).

        Use when you need to label the current place (kitchen, living room, office, etc.).
        Pass a comma-separated list of possible categories.

        Args:
            categories: Comma-separated list of possible categories, e.g. "kitchen, living room, bedroom, office".

        Returns:
            str: The best-matching category and confidence.
        """
        cat_list = [c.strip() for c in categories.split(",") if c.strip()]
        if not cat_list:
            return "Error: Provide at least one category (e.g. 'kitchen, living room')."
        image = vision.capture()
        name, conf = vision.classify_scene(image, cat_list)
        print(f"Classified scene as: {name} (confidence: {conf:.2f})")
        return f"Classified as: {name} (confidence: {conf:.2f})"

    # -------------------------------------------------------------------------
    # Object detection and search
    # -------------------------------------------------------------------------

    @tool
    def detect_objects_from_view() -> str:
        """Detect and list all objects currently visible in the camera view.

        Use when the user asks "what objects do you see?" or "list all items in view".

        Returns:
            str: List of detected objects with their class names and confidence scores.
        """
        CONFIDENCE_THRESHOLD = 0.4
        image = vision.capture()
        objects = vision.detect_objects(image)
        if not objects:
            return "No objects detected in current view."
        descriptions = vision.caption_batch([obj.cropped_image for obj in objects])
        positions = robot.tools.bboxes_to_positions([obj.bbox for obj in objects])
        lines = [f"Detected {len(objects)} object(s):"]
        count = 1
        for obj, desc, position in zip(objects, descriptions, positions):
            class_name = obj.class_name if obj.class_name else "unknown"
            confidence = obj.confidence if obj.confidence else 0.0
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            formatted_pos = f"(x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f})"
            lines.append(f"  - {count}) Class Name: {class_name} Description: {desc} Position: {formatted_pos}")
            count += 1
        # print(f"Detected objects: {lines}")
        print("\n".join(lines))
        return "\n".join(lines)

    @tool
    def find_object_from_view(object_name: str) -> str:
        """Detect a specific object in the current camera view.

        Uses segmentation and embedding similarity to find the best-matching object.
        Use when the user asks "do you see a X?" or "is there a X in view?".

        Args:
            object_name: Name of the object to look for (e.g. "coffee mug", "laptop").

        Returns:
            str: Whether the object was found and a short description, or "not found".
        """
        image = vision.capture()
        query_emb = vision.embed_text(object_name)
        objs = vision.detect_and_embed_objects(image)
        if not objs:
            return f"No objects detected in current view. Cannot confirm '{object_name}'."
        best_sim = -1.0
        best_idx = 0
        for i, o in enumerate(objs):
            sim = vision.embedding.similarity(query_emb, o["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        # Heuristic: require similarity above ~0.25 (normalized CLIP can be 0.2-0.4 for related things)
        if best_sim < 0.22:
            return f"Object '{object_name}' not found in current view. (Best match score: {best_sim:.2f})"
        obj = objs[best_idx]
        print(f"Found '{object_name}' in view (match score: {best_sim:.2f}). Region: bbox={obj['bbox']}, area_ratio={obj['area_ratio']:.2%}.")
        return f"Found '{object_name}' in view (match score: {best_sim:.2f}). Region: bbox={obj['bbox']}, area_ratio={obj['area_ratio']:.2%}."

    @tool
    def find_object_from_memory(object_name: str) -> str:
        """Search the database for where a specific object is or was last seen.

        Use when the user asks "where is the X?" and you need to query stored locations.

        Args:
            object_name: Name of the object to find (e.g. "coffee mug", "fire extinguisher").

        Returns:
            str: Known location(s) and coordinates, or a message if not in database.
        """
        if db is None:
            return "Object database not available. Cannot search for stored objects."
        query_emb = vision.embed_text(object_name)
        hits = db.query_objects(query_emb, n_results=5)
        if not hits:
            return f"No stored location for '{object_name}' in the database."
        lines = [f"Found {len(hits)} location(s) for '{object_name}':"]
        for h in hits:
            xyz = h.get("object_xyz", [0, 0, 0])
            sid = h.get("scene_id", "")
            conf = h.get("similarity", 0)
            lines.append(f"  - position (x={xyz[0]:.2f}, y={xyz[1]:.2f}, z={xyz[2]:.2f}), scene_id={sid}, confidence={conf:.2f}")
        print(f"Found {len(hits)} location(s) for '{object_name}': {lines}")
        return "\n".join(lines)

    @tool
    def find_scene_from_memory(scene_description: str) -> str:
        """Search the database for a scene or location matching the description.

        Use when the user asks "where is the kitchen?" or "find a room with a whiteboard".

        Args:
            scene_description: Description of the scene or location (e.g. "kitchen", "meeting room").

        Returns:
            str: Matching locations and their coordinates, or a message if none found.
        """
        if db is None:
            return "Scene database not available. Cannot search for stored scenes."
        query_emb = vision.embed_text(scene_description)
        hits = db.query_scenes(query_emb, n_results=5)
        if not hits:
            return f"No stored scene matching '{scene_description}' in the database."
        lines = [f"Found {len(hits)} scene(s) matching '{scene_description}':"]
        for h in hits:
            sid = h.get("scene_id", "")
            xyz = h.get("scene_xyz", [0, 0, 0])
            conf = h.get("similarity", 0)
            lines.append(f"  - scene_id={sid}, position (x={xyz[0]:.2f}, y={xyz[1]:.2f}, z={xyz[2]:.2f}), confidence={conf:.2f}")
        print(f"Found {len(hits)} scene(s) matching '{scene_description}': {lines}")
        return "\n".join(lines)

    # @tool
    # def scan_and_remember(
    #     x: float = 0.0,
    #     y: float = 0.0,
    #     z: float = 0.0,
    #     heading: float = 0.0,
    # ) -> str:
    #     """Scan the current view and store detected objects and scene in the database.

    #     Use when the user wants to remember this location or build a map.
    #     If the robot knows its current pose, pass x, y, z, heading so locations are stored correctly.

    #     Args:
    #         x: Current robot x position (meters). Default 0.
    #         y: Current robot y position (meters). Default 0.
    #         z: Current robot z position (meters). Default 0.
    #         heading: Current robot heading (radians). Default 0.

    #     Returns:
    #         str: Summary of what was stored (scene + number of objects).
    #     """
    #     if db is None:
    #         return "Database not available. Cannot store scene or objects."
    #     full_image = vision.capture()
    #     scene_label, _ = vision.classify_scene(
    #         full_image, ["kitchen", "living room", "bedroom", "bathroom", "office", "dining room", "corridor", "other"]
    #     )
    #     scene_id = f"scene_{uuid.uuid4().hex[:8]}"
    #     scene_emb = vision.embed_image(full_image)
    #     db.upsert_scene(
    #         SceneRecord(
    #             scene_id=scene_id,
    #             scene_xyz=[x, y, z],
    #             scene_embedding=scene_emb,
    #             heading=heading,
    #         )
    #     )
    #     objs = vision.detect_and_embed_objects(full_image)
    #     stored = 0
    #     for i, o in enumerate(objs):
    #         print(f"Storing object {i} of {len(objs)}: {o['bbox']}, area_ratio={o['area_ratio']:.2%}")
    #         obj_id = f"obj_{scene_id}_{i}"
    #         db.upsert_object(
    #             ObjectRecord(
    #                 object_id=obj_id,
    #                 object_xyz=[x, y, z],
    #                 object_embedding=o["embedding"],
    #                 heading=heading,
    #                 scene_id=scene_id,
    #                 class_id=o.get("class_id"),
    #                 class_name=o.get("class_name"),
    #             )
    #         )
    #         stored += 1
    #     print(f"Stored scene '{scene_label}' (id={scene_id}) and {stored} object(s) at position (x={x}, y={y}, z={z}), heading={heading}.")
    #     return f"Stored scene '{scene_label}' (id={scene_id}) and {stored} object(s) at position (x={x}, y={y}, z={z}), heading={heading}."

    # -------------------------------------------------------------------------
    # People (caption-based for now)
    # -------------------------------------------------------------------------

    @tool
    def detect_people_from_view() -> str:
        """Detect and describe all people currently visible in the camera view.

        Uses vision model to describe how many people and their approximate poses/positions.
        Use when the user asks "how many people?" or "who do you see?".

        Returns:
            str: Description of people in view (count, poses, positions if possible).
        """
        prompt = (
            "Describe all people visible in this image: how many, their approximate positions (left/center/right, distance), "
            "and what they are doing (standing, sitting, waving, etc.). Be concise."
        )
        print(f"Describing people: {prompt}")
        image = vision.capture()
        return vision.caption(image, prompt=prompt)

    @tool
    def recognize_pose_from_view(person_id: str) -> str:
        """Analyze the pose of a specific person (e.g. person_1, person_2 from detect_people).

        Use after detect_people when the user asks for more detail about one person's pose.

        Args:
            person_id: Identifier like "person_1" or "the person on the left".

        Returns:
            str: Description of that person's pose and posture.
        """
        prompt = (
            f"Focus on {person_id} (or the person referred to). "
            "Describe their body pose in detail: standing/sitting, arm positions, facing direction, gesture if any."
        )
        print(f"Describing pose: {prompt}")
        image = vision.capture()
        return vision.caption(image, prompt=prompt)

    @tool
    def find_person_from_memory(name: str) -> str:
        """Search for a person by name using the database.

        Use when the user asks "where is John?" if you have a people database.
        People database may be empty; returns result of search or message.

        Args:
            name: Name or identifier of the person to find.

        Returns:
            str: Last known location or "not found".
        """
        if db is None:
            return "People database not available. Cannot search for person by name."
        # WalkieVectorDB query_people expects a face_embedding; we don't have one from name.
        # So we can't do semantic search by name without a different index. Return a clear message.
        print(f"Person search by name ('{name}') is not supported yet. The database matches by face embedding. Use detect_people and recognize_face to describe who is in view.")
        return (
            f"Person search by name ('{name}') is not supported yet. "
            "The database matches by face embedding. Use detect_people and recognize_face to describe who is in view."
        )

    return [
        # describe_surroundings_from_view,
        classify_scene_from_view,
        detect_objects_from_view,
        # find_object_from_view,
        find_object_from_memory,
        find_scene_from_memory,
        # scan_and_remember,
        # detect_people_from_view,
        # recognize_pose_from_view,
        # find_person_from_memory,
    ]
