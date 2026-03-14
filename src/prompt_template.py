def earnings_extraction_prompt(document_text, filename):

    return f"""
You are an equity analyst at a hedge fund specialising in researching asset management firms 
(e.g. T. Rowe Price, BlackRock, Franklin Templeton etc).

Your task is to extract structured financial and strategic information from an earnings call transcript.

IMPORTANT RULES

- Extract numeric values where possible. If a number is approximate (e.g. "about", "roughly", "around"),
convert it to the closest numeric value. E.g. "roughly $1.8 trillion" → 1.8
- Return numbers without units. Convert units if necessary. Examples:
            $903 billion → 0.903 trillion
            $1,200 billion → 1.2 trillion
- If a value is not mentioned return null.
- Return ONLY valid JSON.
- Use EXACT key names.
- Example for handling percentages:
        - "up 8%" → 8
        - "8 percent" → 8
- Do not add new fields or rename fields.
- Do not include explanations.
- Before returning null for a field, verify the value does not appear anywhere in the transcript 
(i.e search the whole transcript).
- If multiple values appear, return the most recent full-year value unless the field explicitly refers to Q4.
- Do not infer, unless for company name - see next rule.

Transcript file name:
{filename}

Use the filename as an additional signal to identify the company and earnings quarter.

    Company name rule:

    The company must be one of the following asset managers:
    BlackRock
    T. Rowe Price Group
    Franklin Templeton
    Invesco
    AllianceBernstein
    State Street Global Advisors
    Janus Henderson
    Affiliated Managers Group
    Federated Hermes
    WisdomTree

    For example, if the transcript refers to:

    Janus Henderson Group
    Janus Henderson Group PLC
    JHG

    return the company name as:

    "Janus Henderson"

    Therefore, if the transcript clearly refers to one of these firms, return the exact company name above.

    Look for the company name in:
    - the transcript title
    - the opening paragraph
    - speaker introductions

Earnings date rule:
    The earnings_date is usually the publication date of the transcript.
    Look for phrases like:
    - Published
    - Released
    - Earnings call date      

Transcript excerpt:
"Our assets under management increased to $1.2 trillion, up 8% year-over-year. Net inflows were $5 billion."

Only part (not all of) the expected Output:

{{
"company": "Example Asset Manager",
"earnings_date": "YYYY-MM-DD",
"AUM_total_trillion_usd": 1.2,
"AUM_growth_pct": 8,
"net_flows_billion_usd": 5
}}

Fields to extract:

Company metrics:
- company
- earnings_date (this is the date of the earnings transcript being released (has to be in the format YYYY-MM-DD
and has to be date of earnings transcript or call, return null if not found)
- report_year

AUM metrics:
- AUM_total_trillion_usd
- AUM_growth_pct
            NOTE: AUM may appear as:
            - assets under management
            - AUM
            - client assets
            - total assets managed

Flows:
- net_flows_billion_usd
- net_flows_q4_billion_usd

        NOTE: Net flows may appear as:
        - net inflows
        - net flows
        - organic growth

Flow signals:
- equity_flow_pressure (true/false)
- fixed_income_flow_positive (true/false)
- etf_flow_positive (true/false)

Investment performance:
- funds_outperform_1yr_pct
- funds_outperform_3yr_pct
- funds_outperform_5yr_pct
- funds_outperform_10yr_pct

Fees:
- effective_fee_rate_bps
- fee_rate_trend (increasing, stable, declining)

Expenses:
- operating_expense_growth_guidance_pct_low
- operating_expense_growth_guidance_pct_high

Capital return:
- share_buybacks_million_usd

Strategic initiatives:
List 5 (or up to if less) strategic initiatives mentioned.

Structural industry headwinds:
List 5 risks (or up to if less) or structural challenges.

Management sentiment:
Return a score between -1 (very negative) and +1 (very positive).

Return ONLY this JSON structure:

{{

"company": null,
"earnings_date": null,
"report_year": null,

"AUM_total_trillion_usd": null,
"AUM_growth_pct": null,

"net_flows_billion_usd": null,
"net_flows_q4_billion_usd": null,

"equity_flow_pressure": null,
"fixed_income_flow_positive": null,
"etf_flow_positive": null,

"funds_outperform_1yr_pct": null,
"funds_outperform_3yr_pct": null,
"funds_outperform_5yr_pct": null,
"funds_outperform_10yr_pct": null,

"effective_fee_rate_bps": null,
"fee_rate_trend": NULL,

"operating_expense_growth_guidance_pct_low": null,
"operating_expense_growth_guidance_pct_high": null,

"share_buybacks_million_usd": null,

"strategic_initiatives": [],
"structural_headwinds": [],

"management_sentiment_score": null

}}

Transcript:
----------------
{document_text}
----------------

Return ONLY JSON.
"""