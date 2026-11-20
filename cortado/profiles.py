"""Opponent populations are separate research targets, not difficulty levels."""
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class Profile:
    target: str
    status: str
    objective: str
    training_population: str
    opponent_identity: str
    adaptation: str
    evaluation: str

    def metadata(self):
        return {'schema': 1, **asdict(self)}

PROFILES = {
    'bot': Profile('bot', 'active', 'maximize_long_run_expected_winnings',
                   'self_play', 'versioned_bot_and_checkpoint',
                   'disabled_until_calibrated', 'held_out_bots_and_historical_checkpoints'),
    'human': Profile('human', 'deferred', 'maximize_long_run_expected_winnings',
                     'shared_foundation_then_validated_human_tendency_models',
                     'individual_opponent_and_session', 'disabled_until_calibrated',
                     'held_out_opponents_and_sessions'),
}

def active_profile(target):
    profile = PROFILES[target]
    if profile.status != 'active':
        raise ValueError('Human-opponent development is deferred; use --mode bot. No human model has been trained.')
    return profile
