"""Rule-based eligibility screener.

Takes a simple citizen profile and returns, for every scheme in the corpus,
whether the person is LIKELY eligible / NEEDS CHECK / NOT eligible, with the
reasons. The UI then shows the cited Eligibility passages from the documents,
so the decision is always backed by the source text.

These rules are a simplified screening aid for demonstration, not a legal
determination of eligibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LIKELY, CHECK, NOT = "likely", "check", "not"


@dataclass
class Profile:
    age: int = 30
    gender: str = "female"            # female / male / other
    area: str = "rural"               # rural / urban
    occupation: str = "farmer"        # farmer / artisan / small_business / labourer / salaried / student / other
    owns_farmland: bool = False
    income_tax_payer: bool = False
    government_employee: bool = False
    has_bank_account: bool = True
    low_income_household: bool = False   # BPL / SECC listed / AAY ration card
    kutcha_or_no_house: bool = False
    household_has_lpg: bool = True
    girl_child_under_10: bool = False


@dataclass
class Result:
    scheme_id: str
    status: str
    reasons: list[str] = field(default_factory=list)


def _r(sid, status, *reasons):
    return Result(sid, status, list(reasons))


def screen(p: Profile) -> list[Result]:
    out: list[Result] = []

    # PM-KISAN
    if not p.owns_farmland:
        out.append(_r("pm_kisan", NOT, "Requires cultivable land in the family's name."))
    elif p.income_tax_payer or p.government_employee:
        out.append(_r("pm_kisan", NOT, "Income-tax payers and government employees are excluded."))
    else:
        out.append(_r("pm_kisan", LIKELY, "Landholding farmer family, not in an excluded category."))

    # Ayushman Bharat PM-JAY
    if p.age >= 70:
        out.append(_r("ayushman_bharat_pmjay", LIKELY, "All citizens aged 70+ are covered (Vay Vandana card)."))
    elif p.low_income_household:
        out.append(_r("ayushman_bharat_pmjay", CHECK,
                      "Eligibility depends on SECC 2011 deprivation / occupational criteria — check on beneficiary.nha.gov.in."))
    else:
        out.append(_r("ayushman_bharat_pmjay", NOT, "Not 70+ and not a low-income (SECC) household."))

    # Ujjwala
    if p.gender != "female" or p.age < 18:
        out.append(_r("pm_ujjwala", NOT, "Connection is given to an adult woman (18+) of the household."))
    elif p.household_has_lpg:
        out.append(_r("pm_ujjwala", NOT, "Household must not already have an LPG connection."))
    elif p.low_income_household:
        out.append(_r("pm_ujjwala", LIKELY, "Adult woman from a poor household without LPG."))
    else:
        out.append(_r("pm_ujjwala", CHECK, "Must belong to a listed poor category or give a 14-point declaration."))

    # Sukanya Samriddhi
    out.append(_r("sukanya_samriddhi", LIKELY if p.girl_child_under_10 else NOT,
                  "Account is opened for a girl child below 10 years."))

    # Atal Pension Yojana
    if not (18 <= p.age <= 40):
        out.append(_r("atal_pension", NOT, "Entry age is 18 to 40 years."))
    elif p.income_tax_payer:
        out.append(_r("atal_pension", NOT, "Income-tax payers are not eligible since 1 Oct 2022."))
    elif not p.has_bank_account:
        out.append(_r("atal_pension", CHECK, "Needs a savings bank / post office account."))
    else:
        out.append(_r("atal_pension", LIKELY, "Age 18-40 with a bank account and not a tax payer."))

    # PMJJBY / PMSBY
    if not p.has_bank_account:
        out.append(_r("pmjjby_pmsby", CHECK, "Open a savings bank / post office account first."))
    elif 18 <= p.age <= 50:
        out.append(_r("pmjjby_pmsby", LIKELY, "Eligible for both PMJJBY (18-50) and PMSBY (18-70)."))
    elif 50 < p.age <= 70:
        out.append(_r("pmjjby_pmsby", LIKELY, "Eligible for PMSBY (18-70); PMJJBY entry age is 18-50."))
    else:
        out.append(_r("pmjjby_pmsby", NOT, "Age outside 18-70."))

    # MUDRA
    if p.occupation in {"small_business", "artisan"}:
        out.append(_r("pm_mudra", LIKELY, "Non-farm micro enterprise with credit need up to Rs 20 lakh."))
    else:
        out.append(_r("pm_mudra", CHECK, "Available if you start an income-generating micro business."))

    # MGNREGA
    if p.area == "rural" and p.age >= 18:
        out.append(_r("mgnrega", LIKELY, "Adult member of a rural household willing to do unskilled work."))
    else:
        out.append(_r("mgnrega", NOT, "Only for adult members of rural households."))

    # PM Vishwakarma
    if p.occupation != "artisan":
        out.append(_r("pm_vishwakarma", NOT, "Only for artisans in the 18 listed traditional trades."))
    elif p.government_employee or p.age < 18:
        out.append(_r("pm_vishwakarma", NOT, "Must be 18+ and not in government service."))
    else:
        out.append(_r("pm_vishwakarma", LIKELY, "Self-employed artisan aged 18+."))

    # PMAY-G
    if p.area != "rural":
        out.append(_r("pmay_gramin", NOT, "PMAY-Gramin is for rural households."))
    elif not p.kutcha_or_no_house:
        out.append(_r("pmay_gramin", NOT, "For houseless families or those in kutcha houses."))
    elif p.income_tax_payer or p.government_employee:
        out.append(_r("pmay_gramin", NOT, "Households with tax payers / government employees are excluded."))
    else:
        out.append(_r("pmay_gramin", CHECK if not p.low_income_household else LIKELY,
                      "Selected from the SECC 2011 / Awaas+ Permanent Wait List verified by the Gram Sabha."))

    order = {LIKELY: 0, CHECK: 1, NOT: 2}
    return sorted(out, key=lambda r: order[r.status])
