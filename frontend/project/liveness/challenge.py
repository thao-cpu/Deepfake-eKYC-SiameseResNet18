import random
from dataclasses import dataclass


@dataclass
class Challenge:
    name: str
    target: str


class ChallengeStateMachine:

    def __init__(self):

        self.challenges = [
            Challenge(
                name="blink",
                target="blink"
            ),
            Challenge(
                name="turn_left",
                target="left"
            ),
            Challenge(
                name="turn_right",
                target="right"
            ),
        ]

        self.current = None
        self.completed = False

        self.history = []

    def start(self):

        self.current = random.choice(
            self.challenges
        )

        self.completed = False

        return self.current

    def check(self, features):

        if self.current is None:
            return False

        if self.completed:
            return True

        target = self.current.target

        success = False

        # Blink challenge
        if target == "blink":
            success = features.get(
                "blink",
                False
            )

        # Head turn
        elif target == "left":
            success = (
                features.get("head_direction")
                == "left"
            )

        elif target == "right":
            success = (
                features.get("head_direction")
                == "right"
            )

        if success:

            self.completed = True

            self.history.append(
                self.current.name
            )

        return success

    def reset(self):

        self.current = None
        self.completed = False
        self.history = []

    def is_completed(self):

        return self.completed

    def get_instruction(self):

        if self.current is None:
            return "No challenge"

        if self.current.target == "blink":
            return "Please blink"

        if self.current.target == "left":
            return "Please turn your head left"

        if self.current.target == "right":
            return "Please turn your head right"

        return "Follow the instruction"