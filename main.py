# ============================================================
# SATQUERY AI - FASTAPI BACKEND
# VQA ONLY
# ============================================================

from pathlib import Path
import sys
import shutil
import tempfile

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\SAGESTUDENT1\Desktop\Data"
)

VQA_SRC = (
    PROJECT_ROOT
    / "satquery_stage3_vqa"
    / "src"
)


# ============================================================
# IMPORT PATH
# ============================================================

if str(VQA_SRC) not in sys.path:
    sys.path.insert(0, str(VQA_SRC))


# ============================================================
# MODEL SERVICE
# ============================================================

from vqa_service import vqa_service


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SatQuery AI - VQA API",
    description=(
        "Vision-Language Question Answering API "
        "for remote sensing satellite imagery."
    ),
    version="1.2.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print()
    print("=" * 70)
    print("SATQUERY AI - VQA BACKEND STARTING")
    print("=" * 70)

    print()
    print("[1/1] Loading VQA-10K LoRA service...")

    vqa_service.load()

    print()
    print("=" * 70)
    print("✓ VQA-10K MODEL LOADED")
    print("✓ SATQUERY AI VQA API READY")
    print("=" * 70)
    print()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "SatQuery AI",
        "status": "running",
        "version": "1.2.0",
        "features": [
            "VQA",
        ],
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "vqa_loaded": vqa_service.loaded,
    }


# ============================================================
# VQA
# ============================================================

@app.post("/api/vqa")
async def vqa_endpoint(
    image: UploadFile = File(...),
    question: str = Form(...),
):

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )


    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    filename = (
        image.filename
        or "image.jpg"
    )

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use JPG, PNG, TIFF or WEBP."
            ),
        )


    # --------------------------------------------------------
    # Temporary directory
    # --------------------------------------------------------

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="satquery_vqa_"
        )
    )

    try:

        # ----------------------------------------------------
        # Save uploaded image
        # ----------------------------------------------------

        image_path = (
            temp_dir
            / Path(filename).name
        )

        with image_path.open("wb") as buffer:

            shutil.copyfileobj(
                image.file,
                buffer,
            )


        print()
        print("=" * 70)
        print("VQA REQUEST")
        print("=" * 70)

        print("Image    :", filename)
        print("Question :", question)


        # ----------------------------------------------------
        # Run VQA model
        # ----------------------------------------------------

        result = vqa_service.ask(
            image_path=image_path,
            question=question,
        )


        # ----------------------------------------------------
        # Handle VQA result
        #
        # Current vqa_service returns:
        #
        # {
        #     "answer": "...",
        #     "confidence": 0.67,
        #     "confidence_percent": 66.7,
        #     "confidence_label": "Medium",
        #     "samples_used": 3
        # }
        # ----------------------------------------------------

        if isinstance(result, dict):

            answer = result.get(
                "answer",
                "",
            )

            confidence = result.get(
                "confidence",
                None,
            )

            confidence_percent = result.get(
                "confidence_percent",
                None,
            )

            confidence_label = result.get(
                "confidence_label",
                "Unavailable",
            )

            samples_used = result.get(
                "samples_used",
                None,
            )

        else:

            # Backward compatibility if
            # vqa_service.ask() returns only string

            answer = str(result)

            confidence = None
            confidence_percent = None
            confidence_label = "Unavailable"
            samples_used = None


        # ----------------------------------------------------
        # Validate answer
        # ----------------------------------------------------

        if not answer.strip():

            raise HTTPException(
                status_code=500,
                detail="VQA model returned an empty answer.",
            )


        print()
        print("Answer     :", answer)
        print(
            "Confidence :",
            confidence_percent,
            "%",
        )
        print(
            "Level      :",
            confidence_label,
        )
        print("=" * 70)


        # ----------------------------------------------------
        # API RESPONSE
        # ----------------------------------------------------

        return {
            "success": True,
            "feature": "vqa",
            "question": question,
            "answer": answer,

            "confidence": confidence,
            "confidence_percent": confidence_percent,
            "confidence_label": confidence_label,

            "samples_used": samples_used,
        }


    # --------------------------------------------------------
    # HTTP errors
    # --------------------------------------------------------

    except HTTPException:
        raise


    # --------------------------------------------------------
    # Model / server errors
    # --------------------------------------------------------

    except Exception as e:

        print()
        print("[VQA ERROR]")
        print(str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


# ============================================================
# FEATURES
# ============================================================

@app.get("/api/features")
def features():

    return {
        "features": {

            "vqa": {
                "status": "available",
                "endpoint": "/api/vqa",
                "input": {
                    "image": "JPG / PNG / TIFF / WEBP",
                    "question": "text",
                },
            },

            "captioning": {
                "status": "coming_soon",
            },

            "grounding": {
                "status": "coming_soon",
            },

            "change_vqa": {
                "status": "coming_soon",
            },

            "optical_sar_fusion": {
                "status": "coming_soon",
            },
        }
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )