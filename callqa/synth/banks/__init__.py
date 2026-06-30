"""Domain utterance banks for synthetic call scripts.

Each domain bank is a dict of slots. A slot maps to a list of alternative
lines so generated calls vary. Lines are clean: no real names, no PII, no
account numbers. They are written to sound like real contact-center talk.

Slots used by the generator:
  opening_agent          agent greets and identifies the line
  recording_disclosure   agent says the call may be recorded (compliance)
  verify_request         agent asks the customer to confirm identity
  verify_response        customer gives a confirming detail (no real PII)
  issue_customer         customer states the core problem
  issue_agent            agent acknowledges and starts working it
  back_and_forth_agent   agent mid-call working line
  back_and_forth_cust    customer mid-call line
  escalation_customer    customer asks for a manager or shows strong anger
  escalation_agent       agent responds to an escalation
  closing_agent          agent wraps up and signs off
  closing_customer       customer closing line
"""
from callqa.synth.banks.billing import BILLING
from callqa.synth.banks.cancellation import CANCELLATION
from callqa.synth.banks.refund import REFUND
from callqa.synth.banks.tech_support import TECH_SUPPORT

BANKS = {
    "billing": BILLING,
    "refund": REFUND,
    "cancellation": CANCELLATION,
    "tech_support": TECH_SUPPORT,
}

# Slots every bank must provide.
REQUIRED_SLOTS = (
    "opening_agent",
    "recording_disclosure",
    "verify_request",
    "verify_response",
    "issue_customer",
    "issue_agent",
    "back_and_forth_agent",
    "back_and_forth_cust",
    "escalation_customer",
    "escalation_agent",
    "closing_agent",
    "closing_customer",
)

__all__ = ["BANKS", "REQUIRED_SLOTS", "BILLING", "REFUND", "CANCELLATION", "TECH_SUPPORT"]
