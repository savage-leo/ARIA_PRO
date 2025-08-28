# Add this to main.py after other route imports
from backend.routes import ai_integration

# Add this to the FastAPI app routers list
app.include_router(ai_integration.router)

# Add this to startup_event() after other services start
from backend.services.mt5_ai_integration import mt5_ai_integration
asyncio.create_task(mt5_ai_integration.start())
