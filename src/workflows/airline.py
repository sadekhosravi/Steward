"""The airline policy, transcribed. Every quote is copied, none is remembered.

Read this against `tau2/domains/airline/policy.md` and nothing else. Where the
source is odd, the odd version is here: "2 free checked bag" is singular in the
silver row and plural in the business row, and line 120 reads "the user is should
be refunded" -- both are the document's, and `unquoted` will fail if either is
tidied.

`## Modify flight` becomes four workflows. The section groups four procedures
whose answers for basic economy differ -- flights no, cabin yes, baggage yes,
passengers only in identity -- and the run shows both reviewing agents reading
the first answer as though it governed the rest.

Two things the policy says that are not transcribed as rules anywhere below,
because they are standing rather than procedural, and `core.policy.standing`
already puts them in front of the actor on every turn: one tool call at a time,
and do not volunteer knowledge the tools did not return.
"""

from __future__ import annotations

from workflows import CUSTOMER, Fact, Rule, Workflow

__all__ = [
    "AIRLINE",
    "BOOK",
    "CANCEL",
    "CHANGE_BAGGAGE",
    "CHANGE_CABIN",
    "CHANGE_FLIGHTS",
    "CHANGE_PASSENGERS",
    "COMPENSATE",
    "REPLACE",
    "STANDING",
]

# Sections, spelled as the policy spells them. `applicable` matches on these.
_BOOK = "Book flight"
_MODIFY = "Modify flight"
_CANCEL = "Cancel flight"
_REFUND = "Refunds and Compensation"

# --- the reservation, which four of the six workflows have to read first ------

_USER_ID = Fact(name="the user id", source=CUSTOMER)
_RESERVATION_ID = Fact(
    name="the reservation id", source=f"{CUSTOMER}, or get_user_details if they do not know it"
)
_RESERVATION = Fact(
    name="the reservation: cabin, flights and their status, passengers, "
    "baggages, insurance, created time, payment methods",
    source="get_reservation_details",
)

# The two lines that open both `## Modify flight` and `## Cancel flight`. Same
# words in both sections, so the quote is found either way.
_IDENTIFY = (
    Rule(
        statement="Get the user id and the reservation id before anything else.",
        quote="First, the agent must obtain the user id and reservation id.",
    ),
    Rule(
        statement="The user id has to come from the customer. It cannot be inferred.",
        quote="The user must provide their user id.",
    ),
    Rule(
        statement="A customer who does not know their reservation id is not blocked -- "
        "look it up with get_user_details.",
        quote=(
            "If the user doesn't know their reservation id, the agent should help locate "
            "it using available tools."
        ),
    ),
)


# --- what holds on every turn, whatever is being done -------------------------

STANDING = (
    Rule(
        statement=(
            "Before any write, state what the call will do -- the flights, the cabin, "
            "the figure -- and get the customer to say yes. This applies to booking, "
            "changing flights, changing baggage, changing cabin and changing passengers."
        ),
        quote=(
            "Before taking any actions that update the booking database (booking, "
            "modifying flights, editing baggage, changing cabin class, or updating "
            "passenger information), you must list the action details and obtain "
            "explicit user confirmation (yes) to proceed."
        ),
    ),
    Rule(
        statement="A request the policy forbids is refused, not worked around.",
        quote="You should deny user requests that are against this policy.",
    ),
    Rule(
        statement=(
            "Transfer only when no available action can serve the request -- not "
            "because a rule is hard to check, and not because the customer insists. "
            "Call transfer_to_human_agents first, then send the exact sentence."
        ),
        quote=(
            "You should transfer the user to a human agent if and only if the request "
            "cannot be handled within the scope of your actions. To transfer, first make "
            "a tool call to transfer_to_human_agents, and then send the message 'YOU ARE "
            "BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user."
        ),
    ),
    Rule(
        statement=(
            "basic economy and economy are different cabins. A rule naming one says "
            "nothing about the other."
        ),
        quote=(
            "There are three cabin classes: **basic economy**, **economy**, **business**. "
            "**basic economy** is its own class, completely distinct from **economy**."
        ),
    ),
)


# --- book --------------------------------------------------------------------

BOOK = Workflow(
    name="Book a reservation",
    section=_BOOK,
    facts=(
        _USER_ID,
        Fact(name="trip type, origin, destination, dates", source=CUSTOMER),
        Fact(
            name="the membership level, which sets the free baggage allowance",
            source="get_user_details",
        ),
        Fact(name="the payment methods on the profile", source="get_user_details"),
        Fact(
            name="the flights, their status and the price of the chosen cabin",
            source="search_direct_flight or search_onestop_flight",
        ),
        Fact(name="first name, last name and date of birth of every passenger", source=CUSTOMER),
        Fact(name="how many checked bags are wanted", source=CUSTOMER),
        Fact(name="whether insurance is wanted", source=CUSTOMER),
    ),
    blocks=(
        Rule(
            statement="More than five passengers cannot go on one reservation.",
            quote="Each reservation can have at most five passengers.",
        ),
        Rule(
            statement=(
                "A payment method not already on the customer's profile cannot be used, "
                "whatever they offer."
            ),
            quote="All payment methods must already be in user profile for safety reasons.",
        ),
        Rule(
            statement="A flight that is delayed, on time or flying cannot be booked.",
            quote=(
                "If the status is **delayed** or **on time**, the flight has not taken off, "
                "cannot be booked."
            ),
        ),
        Rule(
            statement="A flight already in the air cannot be booked.",
            quote=(
                "If the status is **flying**, the flight has taken off but not landed, "
                "cannot be booked."
            ),
        ),
    ),
    rules=(
        Rule(
            statement="Ask the customer for their user id; it is the first thing needed.",
            quote="The agent must first obtain the user id from the user.",
        ),
        Rule(
            statement="Ask for trip type, origin and destination before searching.",
            quote="The agent should then ask for the trip type, origin, destination.",
        ),
        Rule(
            statement="One cabin for the whole reservation. Not one cabin per segment.",
            quote="Cabin class must be the same across all the flights in a reservation.",
        ),
        Rule(
            statement=(
                "Every passenger needs all three of first name, last name and date of "
                "birth. A booking is not ready while one is missing."
            ),
            quote=(
                "The agent needs to collect the first name, last name, and date of birth "
                "for each passenger."
            ),
        ),
        Rule(
            statement="One itinerary and one cabin covers everybody on the reservation.",
            quote="All passengers must fly the same flights in the same cabin.",
        ),
        Rule(
            statement=(
                "At most one certificate, one credit card and three gift cards on a "
                "reservation. A fourth gift card, or a second certificate, means the "
                "booking has to be split."
            ),
            quote=(
                "Each reservation can use at most one travel certificate, at most one "
                "credit card, and at most three gift cards."
            ),
        ),
        Rule(
            statement=(
                "Whatever is left on a certificate after the booking is lost, so a "
                "certificate is worth spending on the larger reservation."
            ),
            quote="The remaining amount of a travel certificate is not refundable.",
        ),
        Rule(
            statement=(
                "Free bags depend on membership level and cabin, per passenger. "
                "Anything beyond the free allowance is 50 dollars each, and that "
                "money is part of the total to be paid."
            ),
            quote=(
                "- If the booking user is a regular member:\n"
                "  - 0 free checked bag for each basic economy passenger\n"
                "  - 1 free checked bag for each economy passenger\n"
                "  - 2 free checked bags for each business passenger\n"
                "- If the booking user is a silver member:\n"
                "  - 1 free checked bag for each basic economy passenger\n"
                "  - 2 free checked bag for each economy passenger\n"
                "  - 3 free checked bags for each business passenger\n"
                "- If the booking user is a gold member:\n"
                "  - 2 free checked bag for each basic economy passenger\n"
                "  - 3 free checked bag for each economy passenger\n"
                "  - 4 free checked bags for each business passenger\n"
                "- Each extra baggage is 50 dollars."
            ),
        ),
        Rule(
            statement="Do not add bags nobody asked for.",
            quote="Do not add checked bags that the user does not need.",
        ),
        Rule(
            statement="Ask about insurance. Do not decide it for them.",
            quote="The agent should ask if the user wants to buy the travel insurance.",
        ),
        Rule(
            statement=(
                "Insurance is 30 dollars per passenger, and it is what later makes a "
                "cancellation for health or weather refundable."
            ),
            quote=(
                "The travel insurance is 30 dollars per passenger and enables full refund "
                "if the user needs to cancel the flight given health or weather reasons."
            ),
        ),
    ),
)


# --- modify, as the four procedures it actually is ----------------------------

CHANGE_FLIGHTS = Workflow(
    name="Change the flights on a reservation",
    section=_MODIFY,
    facts=(
        _USER_ID,
        _RESERVATION_ID,
        _RESERVATION,
        Fact(
            name="the replacement flights, their status and their price in the "
            "reservation's existing cabin",
            source="search_direct_flight or search_onestop_flight",
        ),
        Fact(
            name="one gift card or credit card, already on the profile, "
            "for the difference or the refund",
            source=f"{CUSTOMER}, chosen from get_user_details",
        ),
    ),
    blocks=(
        Rule(
            statement=(
                "A basic economy reservation cannot have its flights changed. This rule "
                "is about the flights and nothing else -- it does not stop a cabin "
                "change, a baggage change or a cancellation."
            ),
            quote="Basic economy flights cannot be modified.",
        ),
    ),
    rules=(
        *_IDENTIFY,
        Rule(
            statement=(
                "Origin, destination and trip type stay as they are. Anything else about "
                "the flights may change."
            ),
            quote=(
                "Other reservations can be modified without changing the origin, "
                "destination, and trip type."
            ),
        ),
        Rule(
            statement=(
                "Segments being kept are passed through unchanged and keep the price "
                "already paid; only the new segments are priced at today's fare."
            ),
            quote=(
                "Some flight segments can be kept, but their prices will not be updated "
                "based on the current price."
            ),
        ),
        Rule(
            statement=(
                "A flight change is paid or refunded through exactly one gift card or "
                "credit card, and it has to be one already on the profile."
            ),
            quote=(
                "If the flights are changed, the user needs to provide a single gift card "
                "or credit card for payment or refund method. The payment method must "
                "already be in user profile for safety reasons."
            ),
        ),
        Rule(
            statement=(
                "The tool will accept a call these rules forbid. Check them before "
                "calling it, not after."
            ),
            quote=(
                "The API does not check these for the agent, so the agent must make sure "
                "the rules apply before calling the API!"
            ),
        ),
        Rule(
            statement=(
                "The move is made with update_reservation_flights, passing every segment "
                "the reservation should end up with, not only the ones that change, and "
                "the cabin it already has."
            ),
            quote="Change flights:",
        ),
    ),
)


CHANGE_CABIN = Workflow(
    name="Change the cabin on a reservation",
    section=_MODIFY,
    facts=(
        _USER_ID,
        _RESERVATION_ID,
        _RESERVATION,
        Fact(
            name="the price of the reservation's existing flights in the new cabin",
            source="search_direct_flight or search_onestop_flight",
        ),
        Fact(
            name="one gift card or credit card, already on the profile",
            source=f"{CUSTOMER}, chosen from get_user_details",
        ),
    ),
    blocks=(
        Rule(
            statement="If any flight on the reservation has already been flown, no cabin change.",
            quote=(
                "Cabin cannot be changed if any flight in the reservation has already been flown."
            ),
        ),
    ),
    permits=(
        Rule(
            statement=(
                "Otherwise every reservation can change cabin, basic economy included, "
                "as long as the flights themselves stay the same. A basic economy "
                "reservation being un-modifiable applies to its flights, not to its cabin."
            ),
            quote=(
                "In other cases, all reservations, including basic economy, can change "
                "cabin without changing the flights."
            ),
        ),
    ),
    rules=(
        *_IDENTIFY,
        Rule(
            statement=(
                "A cabin change is made with update_reservation_flights, passing the new "
                "cabin and the flights the reservation already has. There is no separate "
                "cabin tool. This domain has six tools that write, and that is one of "
                "them."
            ),
            quote=(
                "Before taking any actions that update the booking database (booking, "
                "modifying flights, editing baggage, changing cabin class, or updating "
                "passenger information), you must list the action details and obtain "
                "explicit user confirmation (yes) to proceed."
            ),
        ),
        Rule(
            statement=(
                "The new cabin applies to every segment. A cabin change for one leg is "
                "not something the tool can express."
            ),
            quote=(
                "Cabin class must remain the same across all the flights in the same "
                "reservation; changing cabin for just one flight segment is not possible."
            ),
        ),
        Rule(
            statement="An upgrade costs the difference, and the customer pays it.",
            quote=(
                "If the price after cabin change is higher than the original price, the "
                "user is required to pay for the difference."
            ),
        ),
        Rule(
            statement="A downgrade is refunded the difference.",
            quote=(
                "If the price after cabin change is lower than the original price, the "
                "user is should be refunded the difference."
            ),
        ),
    ),
)


CHANGE_BAGGAGE = Workflow(
    name="Change the baggage on a reservation",
    section=_MODIFY,
    facts=(
        _USER_ID,
        _RESERVATION_ID,
        _RESERVATION,
        Fact(
            name="the membership level, which sets the free allowance",
            source="get_user_details",
        ),
        Fact(
            name="one gift card or credit card, already on the profile",
            source=f"{CUSTOMER}, chosen from get_user_details",
        ),
    ),
    blocks=(
        Rule(
            statement=(
                "Bags can be added and never taken away. A request to remove one is "
                "refused, not performed with a lower number."
            ),
            quote="The user can add but not remove checked bags.",
        ),
        Rule(
            statement=(
                "Insurance cannot be added to an existing reservation, at any price, "
                "for any reason."
            ),
            quote="The user cannot add insurance after initial booking.",
        ),
    ),
    rules=(
        *_IDENTIFY,
        Rule(
            statement=(
                "The free allowance is by membership level and cabin, per passenger; "
                "each bag past it is 50 dollars."
            ),
            quote="- Each extra baggage is 50 dollars.",
        ),
        Rule(
            statement=(
                "A baggage change is made with update_reservation_baggages, passing the "
                "total number of checked bags the reservation should end up with and how "
                "many of those are free. There is no add-a-bag tool and nothing has to be "
                "cancelled: the call carries the new total, not the difference."
            ),
            quote="Change baggage and insurance:",
        ),
    ),
)


CHANGE_PASSENGERS = Workflow(
    name="Change the passengers on a reservation",
    section=_MODIFY,
    facts=(
        _USER_ID,
        _RESERVATION_ID,
        _RESERVATION,
        Fact(
            name="the corrected first name, last name and date of birth",
            source=CUSTOMER,
        ),
    ),
    blocks=(
        Rule(
            statement=(
                "The number of passengers cannot change. Who they are can be corrected; "
                "adding or removing one cannot be done."
            ),
            quote="The user can modify passengers but cannot modify the number of passengers.",
        ),
        Rule(
            statement=(
                "Transferring will not help with the number of passengers -- a human "
                "agent cannot do it either, so the answer is no rather than a handoff."
            ),
            quote="Even a human agent cannot modify the number of passengers.",
        ),
    ),
    rules=(
        *_IDENTIFY,
        Rule(
            statement=(
                "Correcting who the passengers are is made with "
                "update_reservation_passengers, passing the whole list of passengers with "
                "the corrections already applied. This is an ordinary change the domain "
                "supports. A misspelled name, a wrong date of birth, or one traveller in "
                "place of another are all done with it, and none of them needs the "
                "reservation cancelled and booked again."
            ),
            quote="Change passengers:",
        ),
    ),
)


# --- replace, which is not a procedure the policy names --------------------

# The one workflow here with no heading of its own in the policy. It exists
# because five separate rules end in the same place -- the change the customer
# wants cannot be made to the reservation they have -- and the policy never says
# what to do next. Three tasks in the last 50x3 lost on exactly that: the
# customer asked for different flights on a basic economy reservation, we called
# `update_reservation_flights`, and the tool took it, because "The API does not
# check these for the agent". Gold cancelled and booked again.
#
# It is filed under Modify because that is the section the customer's request
# lands in. Every quote below is from Modify or Cancel; nothing here is invented.
REPLACE = Workflow(
    name="Replace a reservation: cancel it and book it again",
    section=_MODIFY,
    facts=(
        _USER_ID,
        _RESERVATION_ID,
        _RESERVATION,
        Fact(
            name="the reason for cancelling, and whether the reservation qualifies "
            "to be cancelled at all",
            source=f"{CUSTOMER}, against the reservation",
        ),
        Fact(
            name="the replacement flights and their price in the cabin actually wanted",
            source="search_direct_flight or search_onestop_flight",
        ),
        Fact(
            name="what the new booking will cost and what the old one refunds",
            source="the two prices, subtracted",
        ),
    ),
    # Each of these is a change that cannot be made to the record the customer
    # already holds. Naming the route out is the whole point: read on its own,
    # every one of them reads as a refusal, and that is how they were being read.
    blocks=(
        Rule(
            statement=(
                "Never move the flights on a basic economy reservation by updating "
                "it. Cancel it and book the new itinerary. Changing the cabin in the "
                "same call does not make it lawful -- a cabin change is allowed only "
                "when the flights stay exactly as they are."
            ),
            quote="Basic economy flights cannot be modified.",
        ),
        Rule(
            statement=(
                "Never move the flights and change the cabin on the same reservation "
                "in one update_reservation_flights call. That one tool does both, and "
                "will accept doing them together, but the policy allows a cabin change "
                "only while the flights stay exactly as they are. Cancel and book the "
                "itinerary wanted, in the cabin wanted."
            ),
            quote=(
                "In other cases, all reservations, including basic economy, can change "
                "cabin without changing the flights."
            ),
        ),
        Rule(
            statement=(
                "Never update a reservation to a different origin, destination or "
                "trip type. Flying somewhere else is a new booking: cancel this one "
                "and book that."
            ),
            quote=(
                "Other reservations can be modified without changing the origin, "
                "destination, and trip type."
            ),
        ),
        Rule(
            statement=(
                "Never update a reservation to more or fewer travellers, and never "
                "split one into several by updating it. The count on an existing "
                "record cannot move -- cancel it and book what is wanted. Only the "
                "count. Changing who the travellers are, while the count stays the "
                "same, is update_reservation_passengers and does not come here: one "
                "name in place of another is a correction, not a replacement."
            ),
            quote="The user can modify passengers but cannot modify the number of passengers.",
        ),
        Rule(
            statement=(
                "Never add insurance to a reservation booked without it. Cancel it "
                "and book again with insurance, if that is what the customer wants."
            ),
            quote="The user cannot add insurance after initial booking.",
        ),
    ),
    rules=(
        *_IDENTIFY,
        Rule(
            statement=(
                "The cancellation has to stand on its own grounds. Needing to re-book "
                "is not one of them: if the reservation does not meet one of the four "
                "conditions for cancelling, this route is closed and the customer is "
                "told so."
            ),
            quote="Otherwise, flight can be cancelled if any of the following is true:",
        ),
        Rule(
            statement=(
                "Two writes, in order: cancel_reservation first, then book_reservation. "
                "Plan both. One without the other leaves the customer with nothing, or "
                "with two reservations."
            ),
            quote="The API does not check these for the agent, so the agent must make "
            "sure the rules apply before calling the API!",
        ),
        Rule(
            statement=(
                "The new booking is a booking, not a copy. Cabin, baggage, insurance "
                "and payment are all chosen again from scratch, and the free baggage "
                "allowance is recomputed for the cabin actually booked."
            ),
            quote="Do not add checked bags that the user does not need.",
        ),
        Rule(
            statement=(
                "Say the whole of it before either write: that the change cannot be "
                "made to the existing reservation, what the old one refunds, what the "
                "new one costs, and the difference. It is one decision for the "
                "customer, not two."
            ),
            quote=(
                "Before taking any actions that update the booking database (booking, "
                "modifying flights, editing baggage, changing cabin class, or updating "
                "passenger information), you must list the action details and obtain "
                "explicit user confirmation (yes) to proceed."
            ),
        ),
    ),
)


# --- cancel ------------------------------------------------------------------

CANCEL = Workflow(
    name="Cancel a reservation",
    section=_CANCEL,
    facts=(
        _USER_ID,
        _RESERVATION_ID,
        _RESERVATION,
        Fact(
            name="the reason for cancelling: change of plan, airline cancelled, or other",
            source=CUSTOMER,
        ),
    ),
    blocks=(
        Rule(
            statement=(
                "If any segment has already been flown, cancellation is out of scope "
                "and the customer is transferred."
            ),
            quote=(
                "If any portion of the flight has already been flown, the agent cannot "
                "help and transfer is needed."
            ),
        ),
    ),
    # Four alternatives, one rule each, because that is how they have to be read.
    # Carried as a single rule they render as one long line and stop looking like
    # a choice. Basic economy is not among them -- and neither is anything else: a
    # cancellation refused on a ground that is not on this list is refused on a
    # ground the policy does not have.
    permits=(
        Rule(
            statement="Bought in the last 24 hours, counting from the reservation's created time.",
            quote="- The booking was made within the last 24 hrs",
        ),
        Rule(
            statement="The airline cancelled the flight.",
            quote="- The flight is cancelled by airline",
        ),
        Rule(
            statement="The reservation's cabin is business.",
            quote="- It is a business flight",
        ),
        Rule(
            statement="The reservation has insurance and the reason given is one it covers.",
            quote=(
                "- The user has travel insurance and the reason for cancellation is "
                "covered by insurance."
            ),
        ),
    ),
    rules=(
        *_IDENTIFY,
        Rule(
            statement=(
                "One of the four conditions above is all that is needed. They are "
                "alternatives -- do not look for a second one."
            ),
            quote="Otherwise, flight can be cancelled if any of the following is true:",
        ),
        Rule(
            statement=(
                "Ask why they are cancelling. It is required, and the insurance branch "
                "cannot be decided without it."
            ),
            quote=(
                "The agent must also obtain the reason for cancellation (change of plan, "
                "airline cancelled flight, or other reasons)"
            ),
        ),
        Rule(
            statement=(
                "The tool will cancel a reservation the rules do not permit. Establish "
                "the permitting condition from the reservation record first."
            ),
            quote=(
                "The API does not check that cancellation rules are met, so the agent "
                "must make sure the rules apply before calling the API!"
            ),
        ),
        Rule(
            statement=(
                "Tell them the refund goes back to the original payment methods in "
                "5 to 7 business days."
            ),
            quote="The refund will go to original payment methods within 5 to 7 business days.",
        ),
        Rule(
            statement=(
                "The cancellation itself is made with cancel_reservation, passing the "
                "reservation id and nothing else. It cancels the whole reservation; there "
                "is no way to cancel one leg of one."
            ),
            quote="The refund will go to original payment methods within 5 to 7 business days.",
        ),
    ),
)


# --- compensation ------------------------------------------------------------

COMPENSATE = Workflow(
    name="Offer compensation",
    section=_REFUND,
    facts=(
        _USER_ID,
        _RESERVATION_ID,
        _RESERVATION,
        Fact(
            name="whether the flight was actually cancelled or delayed",
            source="get_flight_status",
        ),
        Fact(name="the membership level", source="get_user_details"),
    ),
    blocks=(
        Rule(
            statement="Never raise compensation first. Only answer a request for it.",
            quote=(
                "Do not proactively offer a compensation unless the user explicitly asks for one."
            ),
        ),
        Rule(
            statement=(
                "A regular member with no insurance flying basic economy or economy gets "
                "nothing. All three have to be true for this to bite."
            ),
            quote=(
                "Do not compensate if the user is regular member and has no travel "
                "insurance and flies (basic) economy."
            ),
        ),
        Rule(
            statement="Cancelled and delayed flights are the only grounds there are.",
            quote="Do not offer compensation for any other reason than the ones listed above.",
        ),
    ),
    permits=(
        Rule(
            statement=(
                "Any ONE of silver or gold membership, insurance, or a business cabin "
                "opens the door. None of them obliges anything on its own."
            ),
            quote=(
                "Only compensate if the user is a silver/gold member or has travel "
                "insurance or flies business."
            ),
        ),
    ),
    rules=(
        Rule(
            statement=(
                "Check the claim against the tools before offering anything. A customer "
                "saying a flight was cancelled is not the flight having been cancelled."
            ),
            quote="Always confirms the facts before offering compensation.",
        ),
        Rule(
            statement=(
                "A certificate is issued with send_certificate, passing the user id and "
                "the amount in dollars. It is the only tool that issues one, and it is "
                "the write this workflow ends in."
            ),
            quote=(
                "If the user complains about cancelled flights in a reservation, the agent "
                "can offer a certificate as a gesture after confirming the facts, with the "
                "amount being $100 times the number of passengers."
            ),
        ),
        Rule(
            statement="A cancelled flight: a certificate for 100 dollars times the passengers.",
            quote=(
                "If the user complains about cancelled flights in a reservation, the agent "
                "can offer a certificate as a gesture after confirming the facts, with the "
                "amount being $100 times the number of passengers."
            ),
        ),
        Rule(
            statement=(
                "A delayed flight: 50 dollars times the passengers, and only once the "
                "change or cancellation they wanted has actually been made."
            ),
            quote=(
                "If the user complains about delayed flights in a reservation and wants to "
                "change or cancel the reservation, the agent can offer a certificate as a "
                "gesture after confirming the facts and changing or cancelling the "
                "reservation, with the amount being $50 times the number of passengers."
            ),
        ),
    ),
)


AIRLINE = (
    BOOK,
    CHANGE_FLIGHTS,
    CHANGE_CABIN,
    CHANGE_BAGGAGE,
    CHANGE_PASSENGERS,
    REPLACE,
    CANCEL,
    COMPENSATE,
)
