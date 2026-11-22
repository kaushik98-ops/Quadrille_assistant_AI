"""Document endpoints: upload a PDF for indexing, and list what is indexed."""
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.datastructures import UploadFile

from assistant.api.deps import Principal, get_principal
from assistant.api.documents import DocumentError

router = APIRouter()

_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]")
SLACK_BYTES = 64 * 1024  # multipart framing allowed on top of the file-size limit


def safe_pdf_name(raw: str | None) -> str | None:
    """Keep only the final path component, replace odd characters, require a .pdf name."""
    base = re.split(r"[\\/]", raw or "")[-1].strip()
    base = _UNSAFE.sub("_", base)
    if not base.lower().endswith(".pdf") or len(base) <= 4:
        return None
    if len(base) > 100:
        base = base[:96] + ".pdf"
    return base[:-4] + ".pdf"


@router.post("/documents")
async def upload_document(request: Request, _: Principal = Depends(get_principal)):
    # No body parameters are declared on purpose: the form is read below, after the gate has
    # already authenticated the request and checked the Content-Length.
    max_mb = request.app.state.cfg.max_upload_mb
    max_bytes = max_mb * 1024 * 1024

    length = request.headers.get("content-length", "")
    if not length.isdigit():
        raise HTTPException(status_code=411, detail="Content-Length header is required.")
    if int(length) > max_bytes + SLACK_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (limit {max_mb} MB).")

    form = await request.form()
    try:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise HTTPException(status_code=400, detail="Send the PDF as a multipart field named 'file'.")
        name = safe_pdf_name(upload.filename)
        if name is None:
            raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")
        data = await upload.read()
    finally:
        await form.close()

    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (limit {max_mb} MB).")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="That file is not a PDF.")

    try:
        chunks = await request.app.state.services.documents.add_pdf(name, data)
    except DocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"filename": name, "chunks": chunks}


@router.get("/documents")
async def list_documents(request: Request, _: Principal = Depends(get_principal)):
    sources = await request.app.state.services.documents.sources()
    return {"documents": [{"name": n, "chunks": c} for n, c in sorted(sources.items())]}
