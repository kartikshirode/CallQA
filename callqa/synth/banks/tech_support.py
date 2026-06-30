"""Technical support utterance bank.

A customer with something broken, and an agent troubleshooting it. Clean and
varied, no PII.
"""

TECH_SUPPORT = {
    "opening_agent": [
        "Thanks for calling tech support, this is Taylor. What's going on?",
        "Hi, technical support here. How can I help you today?",
        "Hello, you've reached support. This is Drew, what can I help with?",
    ],
    "recording_disclosure": [
        "Before we troubleshoot, this call may be recorded for quality and training.",
        "Just so you know, the call is recorded for quality purposes.",
        "Heads up, we record support calls for training.",
    ],
    "verify_request": [
        "Can you confirm the name and email on the account so I can find it?",
        "To look up your setup, what's the email you registered with?",
        "Let me verify you first. What name is the account under?",
    ],
    "verify_response": [
        "Sure, it's my name with the email I always use.",
        "Yeah, the registered email is my usual one and the name matches.",
        "Of course, both are what you've got on file.",
    ],
    "issue_customer": [
        "My internet keeps dropping every few minutes and it's driving me nuts.",
        "The app won't load at all, it just sits on a blank screen.",
        "I can't log in, it says my password is wrong but it isn't.",
    ],
    "issue_agent": [
        "Let's get that sorted. When did this start happening?",
        "Okay, let me walk you through a couple of checks.",
        "Got it. First, can you tell me what you've already tried?",
    ],
    "back_and_forth_agent": [
        "Try restarting the router and give it about a minute to come back.",
        "Let's clear the app cache and reopen it, that fixes most blank screens.",
        "I'm going to send a password reset link, watch for it in a moment.",
    ],
    "back_and_forth_cust": [
        "Okay, hang on, I'm doing that now.",
        "Alright, it's coming back up I think.",
        "Still the same, nothing changed.",
    ],
    "escalation_customer": [
        "I've been on this for an hour. Get me someone who knows what they're doing.",
        "This isn't working. I want to speak to a supervisor.",
        "Forget the steps, escalate me to someone senior right now.",
    ],
    "escalation_agent": [
        "I hear you, this has taken too long. Let me bring in a specialist.",
        "Okay, I'll escalate to our senior tech team so they can dig in.",
        "Sorry this has been rough. Let me get a supervisor on the line.",
    ],
    "closing_agent": [
        "Looks like that fixed it. I'll note the case in case it returns. Anything else?",
        "Glad it's working now. Is there anything else I can help with?",
        "You're back up and running. Thanks for bearing with me.",
    ],
    "closing_customer": [
        "No, that's it. Thanks for sticking with it.",
        "That's all, appreciate the help.",
        "Working now, thank you. Bye.",
    ],
}
