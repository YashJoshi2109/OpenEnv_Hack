"""FastAPI entry point. create_app takes CLASS not instance."""
from openenv.core import create_app
from .negotiation_environment import NegotiationEnvironment
from .models import NegotiationAction, NegotiationObservation

app = create_app(NegotiationEnvironment, NegotiationAction, NegotiationObservation,
                 env_name="negotiation_arena")
