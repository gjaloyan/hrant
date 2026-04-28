"""HTTP API routers, one per concern.

Each module here exports `router: APIRouter`. `backend.main` mounts them
in order. Endpoints keep their original `/api/...` paths so the frontend
contract is unchanged.
"""
