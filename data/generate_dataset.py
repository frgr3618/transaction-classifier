"""Generate a realistically-hard synthetic transaction dataset.

Replaces the old dataset (45 fixed templates -> 100% accuracy) with one that
has a much larger merchant vocabulary, genuine cross-category ambiguity, and
text noise (typos/abbreviations/truncation) inside the merchant description.

Amount/TXN-id suffix format is left untouched so the production clean_text()
regex in src/app.py keeps working unchanged.

Run from the repo root: python data/generate_dataset.py
"""

import random
import secrets
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TRAIN_ROWS = 5000
TEST_ROWS = 1000

OUT_DIR = Path(__file__).parent / "raw"

# ---------------------------------------------------------------------------
# Per-category merchant vocabulary (~30-35 phrases each). A handful of phrases
# are deliberately repeated verbatim across two categories' pools below --
# that's what creates genuine, irreducible label ambiguity (e.g. "Amazon
# payment" can legitimately be shopping or entertainment).
# ---------------------------------------------------------------------------

VOCAB = {
    "education": [
        "College fees payment", "School tuition fee", "University semester fee",
        "Online course subscription", "Udemy course purchase", "Coursera subscription",
        "Byju's learning app fee", "Library fee", "Exam fee payment",
        "Hostel fee payment", "Tuition class fee", "Coaching institute fee",
        "Book store purchase", "Stationery purchase", "Skill development course",
        "Certification exam fee", "Unacademy subscription", "Vedantu tuition fee",
        "School bus fee", "Admission fee payment", "Scholarship application fee",
        "Study material purchase", "Online exam registration", "Distance education fee",
        "School annual fee", "Kindergarten fee payment", "Music class fee",
        "Dance class fee", "Sports academy fee", "Language course fee",
        "Book purchase",
    ],
    "emi": [
        "Personal loan EMI", "Home loan EMI", "Car loan EMI", "Bike loan EMI",
        "Credit card EMI", "HDFC home loan EMI", "SBI car loan EMI",
        "Bajaj Finserv EMI", "Consumer durable loan EMI", "Education loan EMI",
        "Gold loan EMI", "ICICI personal loan EMI", "Axis bank loan EMI",
        "Two wheeler loan EMI", "Mobile phone EMI", "Laptop EMI payment",
        "Appliance EMI payment", "Business loan EMI", "Loan against property EMI",
        "Overdraft EMI payment", "Tractor loan EMI", "Furniture EMI payment",
        "Wedding loan EMI", "Medical loan EMI", "Travel loan EMI",
        "Kotak Mahindra loan EMI", "Yes Bank loan EMI", "IDFC First loan EMI",
        "Flexi loan EMI", "NBFC loan installment",
    ],
    "entertainment": [
        "Netflix subscription", "Hotstar subscription", "Spotify subscription",
        "Movie ticket booking", "BookMyShow ticket", "PVR cinema ticket",
        "INOX movie ticket", "Concert ticket booking", "Gaming purchase",
        "Steam game purchase", "PlayStation store purchase", "YouTube Premium subscription",
        "Sony Liv subscription", "Zee5 subscription", "Comedy show ticket",
        "Amusement park ticket", "Water park entry fee", "Club membership fee",
        "Bowling alley payment", "Escape room booking", "Theatre play ticket",
        "Music streaming subscription", "Gaming console purchase", "Event ticket booking",
        "Adventure park ticket", "Karaoke bar payment", "Online gaming top-up",
        "Amazon payment", "Google Play purchase", "Gym membership fee",
        "Mobile recharge",
    ],
    "food": [
        "Swiggy order", "Zomato order", "Dominos pizza order", "McDonald's meal",
        "KFC order", "Restaurant dining", "Cafe coffee purchase", "Starbucks order",
        "Grocery purchase", "BigBasket order", "Blinkit grocery order",
        "Zepto grocery order", "Big Bazaar purchase", "Reliance Fresh purchase",
        "DMart grocery purchase", "Street food payment", "Bakery purchase",
        "Ice cream parlor payment", "Juice bar payment", "Food court payment",
        "Canteen payment", "Tiffin service payment", "Meal subscription payment",
        "Sweet shop purchase", "Milk delivery payment", "Vegetable vendor payment",
        "Fruit vendor payment", "Dhaba meal payment", "Bar and pub payment",
        "Home delivery food order", "Catering service payment", "Subway sandwich order",
        "Burger King order", "Pizza Hut order", "Cloud kitchen order",
    ],
    "healthcare": [
        "Pharmacy medicine purchase", "Apollo pharmacy purchase", "Hospital consultation fee",
        "Doctor consultation fee", "Dental clinic payment", "Diagnostic lab test fee",
        "Blood test payment", "X-ray scan payment", "MRI scan payment",
        "Health checkup package", "Physiotherapy session fee", "Eye clinic payment",
        "Optical store purchase", "Ayurvedic treatment payment", "Homeopathy clinic payment",
        "Vaccination fee", "Ambulance service payment", "Nursing home payment",
        "Medical store purchase", "Clinic consultation fee", "Skin clinic payment",
        "Fertility clinic payment", "Surgery payment", "Medical equipment purchase",
        "Practo consultation fee", "1mg medicine order", "Netmeds pharmacy order",
        "PharmEasy order", "Gym membership fee", "Insurance premium payment",
    ],
    "investment": [
        "Mutual fund SIP", "Groww ETF investment", "Zerodha stock purchase",
        "Crypto investment", "Fixed deposit investment", "PPF contribution",
        "NPS contribution", "Recurring deposit payment", "Stock market investment",
        "Gold bond investment", "Sovereign gold bond purchase", "Upstox trading investment",
        "ELSS tax saving fund", "Equity mutual fund purchase", "Debt fund investment",
        "Index fund SIP", "Bitcoin purchase", "Ethereum purchase",
        "WazirX crypto purchase", "CoinDCX crypto purchase", "Post office savings scheme",
        "Sukanya Samriddhi contribution", "Bond investment", "Real estate investment",
        "REIT investment", "IPO application payment", "Demat account charges",
        "Portfolio management fee", "Insurance ULIP premium", "Angel One investment",
        "Insurance premium payment",
    ],
    "shopping": [
        "Flipkart order", "Amazon online shopping", "Myntra clothing order",
        "Ajio fashion order", "Nykaa cosmetics order", "IKEA furniture purchase",
        "Reliance Digital purchase", "Croma electronics purchase", "Decathlon sports purchase",
        "Lifestyle store purchase", "Shoppers Stop purchase", "Pantaloons purchase",
        "H&M clothing purchase", "Zara clothing purchase", "Levi's store purchase",
        "Electronics store purchase", "Mobile phone purchase", "Laptop purchase",
        "Watch store purchase", "Jewelry store purchase", "Footwear store purchase",
        "Home decor purchase", "Furniture store purchase", "Toy store purchase",
        "Sportswear purchase", "Cosmetics purchase", "Handbag purchase",
        "Sunglasses purchase", "Perfume store purchase", "Gift shop purchase",
        "Online marketplace order", "Meesho order", "Snapdeal order",
        "Amazon payment", "Google Play purchase", "Book purchase",
    ],
    "travel": [
        "Uber ride", "Ola cab fare", "IndiGo flight ticket", "SpiceJet flight ticket",
        "Air India flight ticket", "IRCTC train ticket", "RedBus bus ticket",
        "MakeMyTrip hotel booking", "Goibibo flight booking", "OYO hotel booking",
        "Airbnb booking payment", "Petrol pump fuel payment", "Diesel fuel payment",
        "Toll payment", "FASTag recharge", "Parking fee payment", "Car rental payment",
        "Bike rental payment", "Metro card recharge", "Auto rickshaw fare",
        "Rapido bike taxi fare", "Cruise booking payment", "Travel insurance payment",
        "Visa application fee", "Passport application fee", "Cab booking payment",
        "Bus ticket booking", "Ferry ticket payment", "Airport lounge fee",
        "Luggage fee payment", "Trekking trip payment", "Resort booking payment",
        "Yatra flight booking", "EaseMyTrip booking", "Taxi fare payment",
    ],
    "utilities": [
        "Electricity bill payment", "Water bill payment", "Gas cylinder booking",
        "Piped gas bill payment", "Internet bill payment", "Broadband recharge",
        "DTH recharge", "Airtel bill payment", "Jio recharge", "Vodafone Idea recharge",
        "BSNL bill payment", "Society maintenance fee", "House rent payment",
        "Property tax payment", "Cable TV bill payment", "Wifi router rental fee",
        "Landline bill payment", "Water purifier service fee", "Waste management fee",
        "Security service fee", "Housekeeping service payment", "Generator fuel payment",
        "Solar panel maintenance fee", "Piped water connection fee", "Municipal tax payment",
        "Newspaper subscription", "Milk subscription payment", "LPG refill payment",
        "Electricity meter fee", "Mobile recharge",
    ],
}

# (min, median, max, sigma) for a lognormal amount sampler, in INR. Shape
# (skew/relative spread) is informed by per-category amount stats pulled from
# the Kaggle "Credit Card Transactions Dataset" by Priyam Choksi
# (https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset,
# 1.3M Sparkov-simulated rows) -- categories with wide variance there
# (travel, shopping, healthcare, investment) get a wider sigma / max than
# tight ones (food, entertainment, utilities, emi). Merchant phrase style is
# partly inspired by the Kaggle "Indian Banking Transaction Text Dataset" by
# coderanand (Apache 2.0). No rows were copied from either dataset.
AMOUNT_PARAMS = {
    "education": (500, 8000, 80000, 0.9),
    "emi": (2000, 12000, 45000, 0.5),
    "entertainment": (100, 500, 5000, 0.6),
    "food": (80, 350, 3000, 0.5),
    "healthcare": (200, 1200, 50000, 0.9),
    "investment": (500, 10000, 100000, 1.0),
    "shopping": (200, 1500, 60000, 0.9),
    "travel": (100, 2500, 80000, 1.0),
    "utilities": (100, 800, 15000, 0.5),
}

ABBREVIATIONS = {
    "payment": "pymt", "subscription": "subscr", "purchase": "purch",
    "booking": "bkng", "restaurant": "restnt", "insurance": "ins",
    "installment": "instl", "delivery": "dlvry", "service": "svc",
    "membership": "membr", "maintenance": "maint", "consultation": "consult",
    "electricity": "electy", "management": "mgmt", "account": "acct",
    "government": "govt", "international": "intl", "professional": "prof",
}

# Generic, brand-free phrases that carry almost no category signal on their
# own -- real bank statements are full of these ("POS purchase", "UPI payment
# to vendor"). Each maps to several *plausible* categories; the same phrase
# gets used verbatim across all of them, so a meaningful chunk of the dataset
# is genuinely ambiguous from text alone, not just noisy. This is what
# actually drives difficulty -- character-level noise on an intact brand
# keyword barely moves a bag-of-words model, since the one maximally
# discriminative token usually survives untouched.
GENERIC_PHRASES = [
    ("Online payment", ["shopping", "entertainment", "utilities", "education"]),
    ("POS purchase", ["shopping", "food", "travel"]),
    ("Merchant payment", ["shopping", "food", "entertainment", "utilities"]),
    ("Card transaction", ["shopping", "food", "travel", "entertainment"]),
    ("UPI payment to vendor", ["shopping", "food", "utilities", "healthcare"]),
    ("Debit card purchase", ["shopping", "food", "travel"]),
    ("Online transaction", ["shopping", "entertainment", "investment", "education"]),
    ("Store purchase", ["shopping", "food"]),
    ("Vendor payment", ["shopping", "food", "healthcare", "utilities"]),
    ("Monthly subscription payment", ["entertainment", "education", "utilities", "investment"]),
    ("Service payment", ["healthcare", "utilities", "travel", "entertainment"]),
    ("Retail purchase", ["shopping", "food"]),
    ("Digital payment", ["shopping", "entertainment", "utilities", "investment"]),
    ("App purchase", ["entertainment", "shopping", "education"]),
    ("Website payment", ["shopping", "entertainment", "education", "travel"]),
    ("Recurring payment", ["emi", "investment", "utilities", "entertainment"]),
    ("Auto debit payment", ["emi", "utilities", "investment"]),
    ("Wallet payment", ["shopping", "food", "travel", "entertainment"]),
    ("NEFT payment", ["emi", "investment", "education", "healthcare"]),
    ("Payment to merchant", ["shopping", "food", "healthcare", "travel"]),
    ("Purchase transaction", ["shopping", "food", "travel"]),
    ("Bill payment", ["utilities", "healthcare", "education", "emi"]),
    ("Subscription payment", ["entertainment", "education", "utilities"]),
    ("Standing instruction payment", ["emi", "investment", "utilities"]),
    ("Direct debit payment", ["emi", "utilities", "investment", "healthcare"]),
]

GENERIC_RATE = 0.25
TYPO_RATE = 0.35
TRUNCATE_RATE = 0.18
SEPARATOR_RATE = 0.15
ABBREV_WORD_RATE = 0.35


def apply_abbreviation(text, rng):
    words = text.split(" ")
    for i, w in enumerate(words):
        key = w.lower().strip(".,")
        if key in ABBREVIATIONS and rng.random() < ABBREV_WORD_RATE:
            words[i] = ABBREVIATIONS[key]
    return " ".join(words)


def apply_typo(text, rng):
    if rng.random() >= TYPO_RATE or len(text) < 6:
        return text
    words = [w for w in text.split(" ") if len(w) >= 4]
    if not words:
        return text
    word = rng.choice(words)
    idx = text.index(word)
    op = rng.choice(["swap", "delete", "duplicate"])
    pos = rng.randrange(1, len(word) - 1) if len(word) > 2 else 0
    if op == "swap" and len(word) > 2:
        chars = list(word)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        new_word = "".join(chars)
    elif op == "delete":
        new_word = word[:pos] + word[pos + 1:]
    else:
        new_word = word[:pos] + word[pos] + word[pos:]
    return text[:idx] + new_word + text[idx + len(word):]


def apply_truncation(text, rng):
    if rng.random() >= TRUNCATE_RATE or len(text) < 16:
        return text
    cut = rng.randint(10, len(text) - 2)
    return text[:cut]


def apply_separator_variance(text, rng):
    if rng.random() >= SEPARATOR_RATE:
        return text
    sep = rng.choice(["-", "*", "/"])
    spaces = [i for i, c in enumerate(text) if c == " "]
    if not spaces:
        return text
    i = rng.choice(spaces)
    return text[:i] + sep + text[i + 1:]


def apply_casing(text, rng):
    style = rng.choice(["upper", "title", "asis"])
    if style == "upper":
        return text.upper()
    if style == "title":
        return text.title()
    return text


def sample_amount(category, rng, np_rng):
    min_v, median_v, max_v, sigma = AMOUNT_PARAMS[category]
    mu = np.log(median_v)
    amount = np_rng.lognormal(mu, sigma)
    amount = max(min_v, min(max_v, amount))
    return int(round(amount))


def _pick_phrase(category, rng):
    if rng.random() < GENERIC_RATE:
        candidates = [p for p, cats in GENERIC_PHRASES if category in cats]
        if candidates:
            return rng.choice(candidates)
    return rng.choice(VOCAB[category])


def generate_row(category, rng, np_rng):
    phrase = _pick_phrase(category, rng)
    phrase = apply_abbreviation(phrase, rng)
    phrase = apply_typo(phrase, rng)
    phrase = apply_truncation(phrase, rng)
    phrase = apply_separator_variance(phrase, rng)
    phrase = apply_casing(phrase, rng)

    amount = sample_amount(category, rng, np_rng)
    txn_id = secrets.token_hex(4)
    text = f"{phrase} INR {amount} TXN{txn_id}"
    return text, category


def generate_dataset(n_rows, rng, np_rng):
    categories = list(VOCAB.keys())
    base = n_rows // len(categories)
    remainder = n_rows - base * len(categories)
    counts = {c: base for c in categories}
    for c in rng.sample(categories, remainder):
        counts[c] += 1

    rows = []
    for category, count in counts.items():
        for _ in range(count):
            rows.append(generate_row(category, rng, np_rng))
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["transaction_text", "category"])


def main():
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    train_df = generate_dataset(TRAIN_ROWS, rng, np_rng)
    test_df = generate_dataset(TEST_ROWS, rng, np_rng)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_DIR / "train_transactions.csv", index=False)
    test_df.to_csv(OUT_DIR / "test_transactions.csv", index=False)

    print(f"Wrote {len(train_df)} train rows and {len(test_df)} test rows to {OUT_DIR}/")
    print("\nCategory counts (train):")
    print(train_df["category"].value_counts())


if __name__ == "__main__":
    main()
