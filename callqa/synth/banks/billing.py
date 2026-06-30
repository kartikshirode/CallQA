"""Billing-dispute utterance bank.

A customer who thinks they were charged wrong, and an agent working through it.
Lines are clean and varied. No real names or account numbers.
"""

BILLING = {
    "opening_agent": [
        "Thanks for calling billing support, my name is Sam. How can I help today?",
        "Hi there, you've reached the billing team. What can I do for you?",
        "Billing support, this is Alex speaking. How can I help you out?",
    ],
    "recording_disclosure": [
        "Quick note before we start, this call may be recorded for quality and training.",
        "Just so you know, this call is recorded for quality purposes.",
        "Heads up, we do record these calls for training and quality.",
    ],
    "verify_request": [
        "Can you confirm the name and the email on the account for me?",
        "To pull up your account, could you give me the billing email please?",
        "I'll need to verify a couple of details first. What's the name on the account?",
    ],
    "verify_response": [
        "Sure, it's under my name and the usual email I sign in with.",
        "Yeah, the email is the one I use for everything, and the name matches.",
        "Of course, name and email are both the ones on file.",
    ],
    "issue_customer": [
        "I got charged twice this month and I only have one subscription.",
        "There's a charge on my card I don't recognize at all.",
        "My bill went up and nobody told me why. It's almost double.",
    ],
    "issue_agent": [
        "Okay, let me pull up the recent charges and take a look.",
        "Got it. Give me a second to check the billing history on this.",
        "I hear you. Let me see what's posted on the account this cycle.",
    ],
    "back_and_forth_agent": [
        "I do see two charges on the same date here, that shouldn't happen.",
        "So one of these looks like a duplicate. I can flag it for a refund.",
        "The price change was a plan adjustment, but I agree it wasn't communicated.",
    ],
    "back_and_forth_cust": [
        "Right, that's exactly what I'm talking about.",
        "So how long is it going to take to fix?",
        "I just want to make sure this doesn't happen again next month.",
    ],
    "escalation_customer": [
        "This is the third time I've called. Put me through to a manager.",
        "I'm done going in circles. I want to speak to a supervisor right now.",
        "No, that's not good enough. Get me someone who can actually fix this.",
    ],
    "escalation_agent": [
        "I understand you're frustrated. Let me bring in my supervisor to help.",
        "That's fair. I'll escalate this to a manager so we can sort it properly.",
        "I'm sorry it's gotten to this point. Let me get a senior agent on the line.",
    ],
    "closing_agent": [
        "I've applied the refund and you'll see it in three to five days. Anything else?",
        "All sorted on my end. Is there anything else I can help with today?",
        "You're all set. Thanks for your patience, and have a good one.",
    ],
    "closing_customer": [
        "No, that's everything. Thanks for the help.",
        "That's all, appreciate it.",
        "Great, thank you. Bye.",
    ],
}
