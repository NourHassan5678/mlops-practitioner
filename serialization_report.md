# Serialization Format Comparison

| Format   | Human-Readable | Cross-Language | Schema-Enforced | Safe from Untrusted Source |
|----------|----------------|----------------|-----------------|----------------------------|
| JSON     | Yes            | Yes            | No              | Yes                        |
| Protobuf | No             | Yes            | Yes             | Yes                        |
| Pickle   | No             | No (Python)    | No              | **No (RCE Vulnerability)** |
| ONNX     | No             | Yes            | Yes             | Yes                        |

**Justification:**
This service serves models using ONNX because it eliminates the arbitrary code execution risks of Pickle while providing cross-language hardware-optimized inference speeds.
