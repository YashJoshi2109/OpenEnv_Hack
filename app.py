"""FastAPI entry point for SalaryNegotiationArena.

Per official slides (page 10):
  from openenv import create_app
  app = create_app(EnvClass, ActionModel, ObsModel, env_name="...")
  
Note: create_app takes the CLASS (factory), not an instance.
"""

from openenv import create_app
from .negotiation_environment import NegotiationEnvironment
from .models import NegotiationAction, NegotiationObservation

app = create_app(
    NegotiationEnvironment,
    NegotiationAction,
    NegotiationObservation,
    env_name="negotiation_arena",
)
