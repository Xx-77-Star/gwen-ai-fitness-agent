# FitLife AI Frontend MVP

Static chat interface for the FitLife AI API. It submits `POST /chat` requests and
renders the returned AI response plus the structured Agent Trace.

## Run

Start the API from the repository root:

```powershell
python -m uvicorn app.main:app --reload
```

Open the API's homepage, or serve this directory for an Avatar-only preview:

```powershell
python -m http.server 8080 --directory frontend
```

Use HTTP rather than `file://` for ES modules and GLB loading. A standalone
static server does not implement `POST /chat`; use the API server for live chat.

The homepage includes the Gwen Hero Avatar, chat, and the developer Trace
drawer. `avatar/AvatarController.js` connects the existing chat busy state to
the shared rendering engine without changing the API. See `avatar/README.md`
for lifecycle, fallback behavior, and homepage browser checks.

## Independent Avatar lab

Serve this directory over HTTP and open `/avatar-test.html`. The viewer is
independent of the homepage UI and uses the same Avatar Engine. See
`AVATAR_VIEWER.md` for animation controls,
automatic framing, model measurements and validation commands.
