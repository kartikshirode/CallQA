"""Refund-request utterance bank.

A customer chasing money back for a return or a bad order, and an agent
processing it. Clean and varied, no PII.
"""

REFUND = {
    "opening_agent": [
        "Thanks for calling, you're through to refunds. My name's Jordan, how can I help?",
        "Hi, refunds team here. What can I sort out for you today?",
        "Hello, this is Riley on the refunds desk. How can I help?",
    ],
    "recording_disclosure": [
        "Before we get into it, this call may be recorded for quality and training.",
        "Just to let you know, the call is recorded for quality purposes.",
        "One thing first, we record these calls for training.",
    ],
    "verify_request": [
        "Can I grab the name and order email so I can find the order?",
        "To look this up, could you confirm the email tied to the order?",
        "Let me verify a couple things first. What name is the order under?",
    ],
    "verify_response": [
        "Yeah, it's under my name with the email I always use.",
        "Sure, the order email is my regular one and the name matches.",
        "Of course, both the name and email are what you've got on file.",
    ],
    "issue_customer": [
        "I returned the item two weeks ago and still haven't seen the refund.",
        "The product showed up broken and I want my money back.",
        "I was charged for something I cancelled before it shipped.",
    ],
    "issue_agent": [
        "Sorry to hear that. Let me check the status of the return.",
        "Okay, let me pull up the order and see where the refund is stuck.",
        "Understood. Give me a moment to look at what happened here.",
    ],
    "back_and_forth_agent": [
        "I can see the return was received, looks like the refund just didn't trigger.",
        "Right, the system held it for review. I can release it now.",
        "Okay, I'll process the full amount back to your original payment method.",
    ],
    "back_and_forth_cust": [
        "So when will it actually land in my account?",
        "That's what I figured. It shouldn't have been this hard.",
        "Okay, as long as it's the full amount.",
    ],
    "escalation_customer": [
        "I've waited long enough. Let me talk to your manager.",
        "This is ridiculous. Get me a supervisor please.",
        "No more excuses. I want someone senior on this call now.",
    ],
    "escalation_agent": [
        "I get it, this has dragged on. Let me loop in my supervisor.",
        "That's reasonable. I'll escalate this so it gets handled faster.",
        "I'm sorry for the wait. Let me bring a manager into the call.",
    ],
    "closing_agent": [
        "Refund's processed, you'll see it within a few business days. Anything else?",
        "All done here. Is there anything else I can help you with?",
        "You're set. Thanks for your patience today.",
    ],
    "closing_customer": [
        "No, that covers it. Thanks.",
        "That's all I needed, cheers.",
        "Perfect, thank you. Bye now.",
    ],
}
