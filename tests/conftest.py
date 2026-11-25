import pytest

from ragpick.parsing import Document

KYC = """1. Introduction
These Directions apply to every regulated entity that opens accounts for customers in India.

2. Record keeping
2.1 Regulated entities shall maintain all necessary records of transactions between the RE and the customer, both domestic and international, for at least five years from the date of transaction.
2.2 Records of identity shall be preserved for at least five years after the business relationship has ended.

3. Periodic updation
3.1 Periodic updation shall be carried out at least once every two years for high risk customers, once every eight years for medium risk customers and once every ten years for low risk customers.
"""

CARDS = """1. Billing
1.1 Card-issuers shall ensure that the customer has at least fifteen days to make payment before interest starts being charged.

2. Closure
2.1 Any request for closure of a credit card shall be honoured within seven working days by the card-issuer.
2.2 Failure to close the card within seven working days shall result in a penalty of Rs 500 per day of delay, as per circular RBI/2022-23/92.
"""


@pytest.fixture
def docs():
    return [Document("kyc_2016", "kyc_2016.txt", KYC, "KYC Direction 2016", {"date": "2016-02-25"}),
            Document("cards_2022", "cards_2022.txt", CARDS, "Credit Card Direction 2022", {"date": "2022-04-21"})]
