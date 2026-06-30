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

The four domain banks used to live in callqa/synth/banks/<domain>.py. They are
small and uniform, so they sit together here behind the same public names
(BANKS, REQUIRED_SLOTS) the generator and tests already import.
"""

# Billing dispute: a customer who thinks they were charged wrong.
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

# Refund request: a customer chasing money back.
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

# Cancellation: a customer trying to close or downgrade.
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

# Tech support: a customer with something not working.
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
