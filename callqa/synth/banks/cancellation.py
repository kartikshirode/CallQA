"""Cancellation and retention utterance bank.

A customer wanting to cancel, and an agent who handles it and sometimes tries
to retain. Clean and varied, no PII.
"""

CANCELLATION = {
    "opening_agent": [
        "Thanks for calling, this is Casey in account services. How can I help?",
        "Hi, you've reached the cancellations team. What can I do for you?",
        "Hello, this is Morgan. How can I help with your account today?",
    ],
    "recording_disclosure": [
        "Before we start, this call may be recorded for quality and training.",
        "Quick note, the call is recorded for quality purposes.",
        "Just so you're aware, we record these calls for training.",
    ],
    "verify_request": [
        "Can you confirm the name and email on the account first?",
        "To make changes I'll need to verify you. What's the account email?",
        "Let me confirm a couple details. What name is the account under?",
    ],
    "verify_response": [
        "Sure, it's my name and the email I sign in with.",
        "Yeah, the email's my usual one and the name matches.",
        "Of course, both are what you have on file.",
    ],
    "issue_customer": [
        "I want to cancel my subscription, I'm not using it anymore.",
        "I'd like to close my account effective today.",
        "I'm moving to a different service, so I need to cancel.",
    ],
    "issue_agent": [
        "Sorry to hear that. Let me pull up your plan and walk you through it.",
        "Okay, I can help with that. Let me check what's on the account.",
        "No problem. Before I do, let me see what plan you're on.",
    ],
    "back_and_forth_agent": [
        "I can cancel today, just so you know it stays active until the cycle ends.",
        "There's a loyalty discount I could apply if you'd consider staying.",
        "Understood. I'll set the cancellation to process at the end of the period.",
    ],
    "back_and_forth_cust": [
        "No, I've made up my mind, I just want it cancelled.",
        "How much would that discount actually be?",
        "Okay, so I won't be charged again after this, right?",
    ],
    "escalation_customer": [
        "I asked to cancel, not for a sales pitch. Get me a manager.",
        "Stop trying to keep me. I want a supervisor now.",
        "This is exactly why I'm leaving. Put someone senior on.",
    ],
    "escalation_agent": [
        "Understood, I won't push it. Let me bring in my supervisor.",
        "Fair enough. I'll escalate so a manager can finalize this.",
        "I'm sorry, that wasn't my intent. Let me get a senior agent for you.",
    ],
    "closing_agent": [
        "Your cancellation is confirmed for the end of the cycle. Anything else?",
        "All set, the account will close as requested. Anything else I can do?",
        "Done. Sorry to see you go, and thanks for being with us.",
    ],
    "closing_customer": [
        "No, that's it. Thanks.",
        "That's everything, appreciate it.",
        "Good, thank you. Bye.",
    ],
}
