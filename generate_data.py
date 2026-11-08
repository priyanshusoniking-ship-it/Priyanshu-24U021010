"""Data generator for Personalized News Feed project."""
import json
import pandas as pd

ARTICLES = [
    # AI & Technology
    {
        "news_id": "N001",
        "category": "AI & Technology",
        "subcategory": "Generative AI",
        "title": "Breakthrough in Multi-Modal Reasoning Models Sets New Reasoning Benchmarks",
        "abstract": "Next-generation multi-modal transformers demonstrate human-level mathematical deduction and image comprehension with 40% reduced inference latency.",
        "published_hours_ago": 2.5,
        "popularity_score": 8450,
        "reading_time_min": 4
    },
    {
        "news_id": "N002",
        "category": "AI & Technology",
        "subcategory": "Semiconductors",
        "title": "Next-Gen 2nm Fabrication Nodes Enter Volume Production for AI Accelerators",
        "abstract": "Leading semiconductor foundries confirm commercial wafer delivery, promising significant energy efficiency gains for high-throughput data centers.",
        "published_hours_ago": 5.0,
        "popularity_score": 6200,
        "reading_time_min": 5
    },
    {
        "news_id": "N003",
        "category": "AI & Technology",
        "subcategory": "Open Source AI",
        "title": "Open Source Small Language Models Outperform Legacy Systems on Edge Devices",
        "abstract": "Compact quantized architectures achieve high precision on localized hardware, unlocking private offline assistants for smartphones and robotics.",
        "published_hours_ago": 9.0,
        "popularity_score": 5300,
        "reading_time_min": 3
    },
    {
        "news_id": "N004",
        "category": "AI & Technology",
        "subcategory": "Autonomous Systems",
        "title": "Robotaxi Fleets Expand Commercial Operations Across Major Metropolitan Corridors",
        "abstract": "Autonomous mobility providers receive regulatory approval for 24/7 passenger transit following 20 million incident-free driverless test miles.",
        "published_hours_ago": 18.0,
        "popularity_score": 7100,
        "reading_time_min": 6
    },
    {
        "news_id": "N005",
        "category": "AI & Technology",
        "subcategory": "Cybersecurity",
        "title": "Zero-Trust Security Architectures Adopt Neural Anomaly Detection to Combat Deepfakes",
        "abstract": "Enterprise security frameworks integrate transformer-driven cryptographic verification to authenticate streaming executive media and biometric logins.",
        "published_hours_ago": 28.0,
        "popularity_score": 3800,
        "reading_time_min": 4
    },
    {
        "news_id": "N006",
        "category": "AI & Technology",
        "subcategory": "Quantum Computing",
        "title": "Logical Qubit Error Correction Demonstration Surpasses Physical Threshold",
        "abstract": "Researchers demonstrate real-time syndrome measurement that suppresses quantum decoherence, bringing fault-tolerant quantum algorithms closer.",
        "published_hours_ago": 36.0,
        "popularity_score": 4900,
        "reading_time_min": 7
    },

    # Climate & Clean Energy
    {
        "news_id": "N007",
        "category": "Climate & Clean Energy",
        "subcategory": "Solar Energy",
        "title": "Perovskite-Silicon Tandem Solar Cells Exceed 34% Practical Power Conversion",
        "abstract": "New commercial solar modules shatter efficiency ceilings, dramatically lowering rooftop generation costs across global residential markets.",
        "published_hours_ago": 3.0,
        "popularity_score": 5900,
        "reading_time_min": 4
    },
    {
        "news_id": "N008",
        "category": "Climate & Clean Energy",
        "subcategory": "Battery Tech",
        "title": "Solid-State Battery Production Pilot Scales to 1,000 km Electric Vehicle Range",
        "abstract": "Automakers unveil non-flammable solid electrolyte packs that achieve 80% recharge in under ten minutes with zero thermal runaway risks.",
        "published_hours_ago": 8.5,
        "popularity_score": 9100,
        "reading_time_min": 5
    },
    {
        "news_id": "N009",
        "category": "Climate & Clean Energy",
        "subcategory": "Carbon Capture",
        "title": "Direct Air Carbon Mineralization Plant Captures 500,000 Metric Tons Annually",
        "abstract": "Geothermal-powered negative-emissions facility permanently stores atmospheric CO2 into basaltic geological formations within two years.",
        "published_hours_ago": 14.0,
        "popularity_score": 3400,
        "reading_time_min": 4
    },
    {
        "news_id": "N010",
        "category": "Climate & Clean Energy",
        "subcategory": "Grid Storage",
        "title": "Sodium-Ion Energy Storage Farms Decouple Renewable Grids from Lithium Supply",
        "abstract": "Abundant mineral chemistries provide affordable 12-hour continuous grid balancing, accelerating the phase-out of natural gas peaker plants.",
        "published_hours_ago": 26.0,
        "popularity_score": 4200,
        "reading_time_min": 5
    },
    {
        "news_id": "N011",
        "category": "Climate & Clean Energy",
        "subcategory": "Green Hydrogen",
        "title": "Offshore Wind Electrolysis Facility Produces First Industrial Green Hydrogen",
        "abstract": "Heavy industrial steelmakers transition furnaces from metallurgical coke to zero-carbon green hydrogen produced directly at sea.",
        "published_hours_ago": 42.0,
        "popularity_score": 2900,
        "reading_time_min": 6
    },

    # Business & Global Markets
    {
        "news_id": "N012",
        "category": "Business & Global Markets",
        "subcategory": "Central Banks",
        "title": "Global Central Banks Signal Coordinated Shift Toward Neutral Interest Rates",
        "abstract": "Cooling inflation prints and resilient employment prompt monetary policy easing, boosting equity valuations and emerging market bond inflows.",
        "published_hours_ago": 1.5,
        "popularity_score": 8800,
        "reading_time_min": 4
    },
    {
        "news_id": "N013",
        "category": "Business & Global Markets",
        "subcategory": "Venture Capital",
        "title": "Venture Capital Inflows Pivot Strongly Toward Physical AI and Robotics Hardware",
        "abstract": "Early-stage valuations surge for embodied AI startups building humanoids, agricultural automation, and automated supply chain logistics.",
        "published_hours_ago": 7.0,
        "popularity_score": 6700,
        "reading_time_min": 5
    },
    {
        "news_id": "N014",
        "category": "Business & Global Markets",
        "subcategory": "Commodities",
        "title": "Critical Mineral Trade Agreements Reshape Global EV and Turbine Supply Chains",
        "abstract": "Bilateral extraction partnerships streamline lithium, cobalt, and rare-earth processing while insulating manufacturers from regional export curbs.",
        "published_hours_ago": 16.0,
        "popularity_score": 3900,
        "reading_time_min": 5
    },
    {
        "news_id": "N015",
        "category": "Business & Global Markets",
        "subcategory": "Fintech",
        "title": "Cross-Border Instant Settlement Protocols Bypass Traditional Nostro Accounts",
        "abstract": "Central bank digital currency pilots execute real-time gross settlement between financial hubs, cutting foreign exchange transaction fees by 65%.",
        "published_hours_ago": 22.0,
        "popularity_score": 4600,
        "reading_time_min": 4
    },
    {
        "news_id": "N016",
        "category": "Business & Global Markets",
        "subcategory": "Retail & E-commerce",
        "title": "AI-Powered Predictive Logistics Reduces Unsold Inventory Waste by One-Third",
        "abstract": "Global omni-channel retailers deploy hyper-local demand forecasting models, cutting return logistics carbon footprints and markdowns.",
        "published_hours_ago": 38.0,
        "popularity_score": 3100,
        "reading_time_min": 3
    },

    # Science & Space Exploration
    {
        "news_id": "N017",
        "category": "Science & Space",
        "subcategory": "Astronomy",
        "title": "Deep Space Telescope Detects Atmospheric Biosignature Candidates on Habitable Exoplanet",
        "abstract": "Spectroscopic analysis reveals methane and carbon dioxide balance in an Earth-mass rocky world orbiting within its parent star's temperate zone.",
        "published_hours_ago": 4.5,
        "popularity_score": 9600,
        "reading_time_min": 6
    },
    {
        "news_id": "N018",
        "category": "Science & Space",
        "subcategory": "Lunar Exploration",
        "title": "Lunar South Pole Mission Uncovers Subsurface Water Ice Deposits in Permanent Shadow",
        "abstract": "Autonomous rovers extract subsurface core samples confirming high-purity water ice vital for sustained life support and rocket propellants.",
        "published_hours_ago": 11.0,
        "popularity_score": 7500,
        "reading_time_min": 5
    },
    {
        "news_id": "N019",
        "category": "Science & Space",
        "subcategory": "Particle Physics",
        "title": "High-Luminosity Collider Upgrade Observes Rare Higgs Boson Decay Channels",
        "abstract": "Precision measurement of Higgs coupling to second-generation fermions confirms Standard Model predictions while constraining supersymmetric physics.",
        "published_hours_ago": 25.0,
        "popularity_score": 4100,
        "reading_time_min": 7
    },
    {
        "news_id": "N020",
        "category": "Science & Space",
        "subcategory": "Fusion Energy",
        "title": "Magnetic Confinement Tokamak Sustains High-Beta Plasma for Record 1,000 Seconds",
        "abstract": "Superconducting magnet arrays maintain steady-state fusion conditions without wall disruptions, proving key milestone for commercial power reactors.",
        "published_hours_ago": 34.0,
        "popularity_score": 8300,
        "reading_time_min": 5
    },

    # Health & Medicine
    {
        "news_id": "N021",
        "category": "Health & Medicine",
        "subcategory": "Genomics",
        "title": "In Vivo CRISPR Epigenome Editing Halts Hereditary Cardiovascular Disease",
        "abstract": "Single-infusion targeted lipid nanoparticles switch off mutant hepatic genes, reducing pathogenic cholesterol biomarkers by 80% for lifetime protection.",
        "published_hours_ago": 6.0,
        "popularity_score": 7800,
        "reading_time_min": 5
    },
    {
        "news_id": "N022",
        "category": "Health & Medicine",
        "subcategory": "Neuroscience",
        "title": "Non-Invasive Neural Interface Decodes Speech at 120 Words Per Minute",
        "abstract": "High-density optical sensors and neural language decoders restore fluid conversational ability for patients suffering from motor paralysis.",
        "published_hours_ago": 13.0,
        "popularity_score": 8200,
        "reading_time_min": 6
    },
    {
        "news_id": "N023",
        "category": "Health & Medicine",
        "subcategory": "Oncology",
        "title": "Universal mRNA Cancer Vaccine Demonstrates Prolonged Remission in Phase III Trials",
        "abstract": "Personalized neoantigen vaccines combined with checkpoint blockade therapy eliminate micro-metastases across aggressive pancreatic malignancies.",
        "published_hours_ago": 20.0,
        "popularity_score": 9400,
        "reading_time_min": 6
    },
    {
        "news_id": "N024",
        "category": "Health & Medicine",
        "subcategory": "Longevity",
        "title": "Targeted Senolytic Clearance Protocol Reverses Cellular Aging Markers in Primate Models",
        "abstract": "Clearing senescent 'zombie' cells rejuvenates musculoskeletal density and cognitive flexibility, paving the way for human clinical longevity trials.",
        "published_hours_ago": 32.0,
        "popularity_score": 6900,
        "reading_time_min": 5
    },

    # Sports & Athletics
    {
        "news_id": "N025",
        "category": "Sports & Athletics",
        "subcategory": "Football / Soccer",
        "title": "Tactical Analytics Overhaul: How Real-Time Biometric Tracking Transformed the Champions League",
        "abstract": "Elite football clubs employ computer vision and spatial pressure heatmaps to execute pressing traps and predict player hamstring fatigue.",
        "published_hours_ago": 8.0,
        "popularity_score": 6400,
        "reading_time_min": 4
    },
    {
        "news_id": "N026",
        "category": "Sports & Athletics",
        "subcategory": "Motorsport",
        "title": "Electric Racing Series Introduces Dynamic 600kW Flash Boost Recharging Stops",
        "abstract": "Revolutionary ultra-fast pit stop charging allows open-wheel electric racecars to run lighter batteries with increased straight-line speed.",
        "published_hours_ago": 19.0,
        "popularity_score": 5100,
        "reading_time_min": 4
    },
    {
        "news_id": "N027",
        "category": "Sports & Athletics",
        "subcategory": "Athletics",
        "title": "Biomechanically Engineered Spikes and Energy-Return Tracks Propel World Record Assaults",
        "abstract": "Advanced carbon-foam geometries and tuned stadium surfaces spark intense debates over technological doping in Olympic sprint events.",
        "published_hours_ago": 29.0,
        "popularity_score": 4500,
        "reading_time_min": 4
    },

    # Global Affairs & Society
    {
        "news_id": "N028",
        "category": "Global Affairs",
        "subcategory": "Diplomacy",
        "title": "International AI Governance Accord Signed by 40 Nations at Geneva Summit",
        "abstract": "World leaders establish legally binding standards for safe model deployment, red-teaming verification, and global compute monitoring.",
        "published_hours_ago": 3.5,
        "popularity_score": 7900,
        "reading_time_min": 5
    },
    {
        "news_id": "N029",
        "category": "Global Affairs",
        "subcategory": "Smart Cities",
        "title": "Autonomous Green Transit Corridors Eliminate Traffic Congestion in Scandinavian Capitals",
        "abstract": "Sensor-integrated bus rapid transit systems powered by micro-hydro grids achieve zero gridlock while converting former asphalt lanes to parks.",
        "published_hours_ago": 15.0,
        "popularity_score": 5500,
        "reading_time_min": 4
    },
    {
        "news_id": "N030",
        "category": "Global Affairs",
        "subcategory": "Education",
        "title": "Adaptive Socratic AI Tutors Narrow Learning Gaps Across Rural School Districts",
        "abstract": "Personalized real-time pedagogical assistants boost STEM comprehension scores by two standard deviations among historically underserved students.",
        "published_hours_ago": 27.0,
        "popularity_score": 6100,
        "reading_time_min": 5
    }
]

USER_PROFILES = [
    {
        "user_id": "U101",
        "name": "Aarav Sharma",
        "persona": "AI & Tech Engineer",
        "description": "Obsessed with frontier language models, semiconductor breakthroughs, autonomous systems, and edge computing.",
        "history": ["N001", "N002", "N003", "N004"],
        "target_positive": "N005"
    },
    {
        "user_id": "U102",
        "name": "Dr. Elena Rostova",
        "persona": "Biomedical & Biotech Researcher",
        "description": "Follows gene therapy, CRISPR advances, non-invasive neurotechnology, oncology, and cellular rejuvenation.",
        "history": ["N021", "N022", "N023"],
        "target_positive": "N024"
    },
    {
        "user_id": "U103",
        "name": "Marcus Vance",
        "persona": "Macro Investor & Clean Energy Analyst",
        "description": "Tracks central bank policy, battery scaling, grid storage, and critical mineral supply chains.",
        "history": ["N008", "N010", "N012", "N013"],
        "target_positive": "N014"
    },
    {
        "user_id": "U104",
        "name": "Samantha Chen",
        "persona": "Astrophysics & Space Enthusiast",
        "description": "Passionate about exoplanet discovery, lunar exploration, quantum physics, and nuclear fusion.",
        "history": ["N017", "N018", "N020"],
        "target_positive": "N019"
    },
    {
        "user_id": "U105",
        "name": "David O'Connor",
        "persona": "Sports & Sports-Tech Fan",
        "description": "Follows modern sports analytics, biometric tracking, EV racing, and biomechanical athletic gear.",
        "history": ["N025", "N026"],
        "target_positive": "N027"
    }
]

def main():
    df = pd.DataFrame(ARTICLES)
    df.to_csv("data/news_catalog.csv", index=False)
    print(f"Saved {len(df)} news articles to data/news_catalog.csv")

    with open("data/user_profiles.json", "w") as f:
        json.dump(USER_PROFILES, f, indent=2)
    print(f"Saved {len(USER_PROFILES)} user personas to data/user_profiles.json")

if __name__ == "__main__":
    main()
