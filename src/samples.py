"""Three short articles for a live demonstration.

Each one starts with a sentence the classroom grammar can parse, and each
one uses vocabulary typical of one BBC category. They are not copied from
the dataset.
"""

SAMPLES = [
    {
        "id": "technology",
        "label": "Technology sample",
        "expect": "Technology",
        "text": (
            "The company launched a new chip. "
            "The company launched a new chip designed for artificial intelligence software "
            "used in cloud computing. Engineers said the processor helps computer systems "
            "train digital models faster while using less power. Technology firms and internet "
            "platforms welcomed the breakthrough, and developers can download a software kit "
            "from the company website. Several companies are running trials of these technologies "
            "in hospitals and research labs. Users will reach the tools through an online service. "
            "Analysts said demand for specialised computer hardware should keep growing as digital "
            "services expand. Investors described the launch as a strong success for the firm's "
            "research team and a positive step for the wider technology market."
        ),
    },
    {
        "id": "sports",
        "label": "Sports sample",
        "expect": "Sports",
        "text": (
            "The striker scored a late goal. "
            "Manchester United secured a vital football victory on Saturday after the striker "
            "scored a late goal against their closest league rivals. The win keeps the club at "
            "the top of the table and strengthens their champion hopes this season. Players "
            "celebrated with the coach as supporters filled the stadium. The captain said the "
            "team had controlled the match after half time, and the goalkeeper made two important "
            "saves. It was a brilliant team performance in a difficult away game. The club now "
            "looks ready for the cup fixture next week. Several players are running extra training "
            "sessions before the next match."
        ),
    },
    {
        "id": "politics",
        "label": "Politics sample",
        "expect": "Politics",
        "text": (
            "The minister faced a fierce protest. "
            "The minister faced a fierce protest outside parliament after the government announced "
            "a controversial tax bill. Opposition leaders said the party had broken its election "
            "promise, and angry campaigners called for a vote of no confidence. Members of "
            "parliament will debate the bill in the commons next week. Critics argued the policy "
            "would hurt public services, while the prime minister defended the plan. The secretary "
            "told reporters that the government would not withdraw the legislation despite the "
            "growing backlash. Unions warned of a political crisis if ministers ignored the protest."
        ),
    },
]
