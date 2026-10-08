"""Retain the four Avatar clips and compact their unused binary data.

Usage: python tools/prune-avatar-animations.py SOURCE.glb OUTPUT.glb
Geometry, skeleton, materials, textures and retained keyframes stay byte-exact.
"""
import argparse
import json
from pathlib import Path
import struct

KEEP = ("Idle_Base", "Interact", "Run", "Cast_Cycle")
COMPONENT_BYTES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
ELEMENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def prune(source: Path, destination: Path) -> dict:
    data = source.read_bytes()
    magic, version, total = struct.unpack_from("<III", data)
    assert magic == 0x46546C67 and version == 2 and total == len(data), "Invalid GLB"
    json_length, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == 0x4E4F534A
    document = json.loads(data[20:20 + json_length])
    bin_length, bin_type = struct.unpack_from("<II", data, 20 + json_length)
    assert bin_type == 0x004E4942 and 28 + json_length + bin_length == len(data)
    binary = data[28 + json_length:]
    assert len(document["buffers"]) == 1
    assert set(document.get("extensionsUsed", [])) <= {"KHR_materials_unlit"}
    clips = {clip["name"]: clip for clip in document["animations"]}
    assert all(name in clips for name in KEEP), "Required animation is missing"
    selected = [clips[name] for name in KEEP]
    references = lambda animations: {
        sampler[key]
        for clip in animations for sampler in clip["samplers"] for key in ("input", "output")
    }
    all_animation_accessors = references(document["animations"])
    retained_animation_accessors = references(selected)
    accessors = document["accessors"]
    assert all("sparse" not in a for a in accessors), "Sparse accessors need a different packer"
    retained = sorted((set(range(len(accessors))) - all_animation_accessors) | retained_animation_accessors)
    accessor_map = {old: new for new, old in enumerate(retained)}
    views = document["bufferViews"]
    rebuilt = bytearray()
    for view_index, view in enumerate(views):
        start = view.get("byteOffset", 0)
        payload = binary[start:start + view["byteLength"]]
        users = {i for i, a in enumerate(accessors) if a.get("bufferView") == view_index}
        is_image = any(image.get("bufferView") == view_index for image in document.get("images", []))
        if users and users <= all_animation_accessors and not is_image:
            assert "byteStride" not in view
            compact = bytearray()
            for index in sorted(users & retained_animation_accessors):
                accessor = accessors[index]
                length = accessor["count"] * COMPONENT_BYTES[accessor["componentType"]] * ELEMENTS[accessor["type"]]
                old_offset = accessor.get("byteOffset", 0)
                compact.extend(b"\0" * (-len(compact) % 4))
                accessor["byteOffset"] = len(compact)
                compact.extend(payload[old_offset:old_offset + length])
            payload = compact
        rebuilt.extend(b"\0" * (-len(rebuilt) % 4))
        view["byteOffset"] = len(rebuilt)
        view["byteLength"] = len(payload)
        rebuilt.extend(payload)
    for mesh in document["meshes"]:
        for primitive in mesh["primitives"]:
            primitive["attributes"] = {k: accessor_map[v] for k, v in primitive["attributes"].items()}
            if "indices" in primitive:
                primitive["indices"] = accessor_map[primitive["indices"]]
            for target in primitive.get("targets", []):
                for name, value in target.items():
                    target[name] = accessor_map[value]
    for skin in document.get("skins", []):
        if "inverseBindMatrices" in skin:
            skin["inverseBindMatrices"] = accessor_map[skin["inverseBindMatrices"]]
    for clip in selected:
        for sampler in clip["samplers"]:
            for key in ("input", "output"):
                sampler[key] = accessor_map[sampler[key]]
    document["animations"] = selected
    document["accessors"] = [accessors[i] for i in retained]
    document["buffers"][0]["byteLength"] = len(rebuilt)
    encoded = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    rebuilt.extend(b"\0" * (-len(rebuilt) % 4))
    output = (
        struct.pack("<III", magic, version, 28 + len(encoded) + len(rebuilt))
        + struct.pack("<II", len(encoded), json_type) + encoded
        + struct.pack("<II", len(rebuilt), bin_type) + rebuilt
    )
    destination.write_bytes(output)
    return {"retained": list(KEEP), "removed": len(clips) - len(KEEP), "before_bytes": len(data), "after_bytes": len(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(prune(args.source, args.destination), indent=2))
