# Import libraries
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
import random
import re
import warnings

os.makedirs("yearly_networks", exist_ok=True)
warnings.filterwarnings("ignore")

# Part 1: Data Preparation


def load_data():
    """Load data from database."""
    connection_string = "mysql+pymysql://root@localhost:3306/RAMAPO"

    engine = create_engine(connection_string)
    query = """
    SELECT
        omek_search_texts.title,
        omek_tags.name AS tag_name,
        element_texts.text AS date
    FROM
        omek_items
    JOIN
        omek_item_types ON omek_items.item_type_id = omek_item_types.id
    JOIN
        omek_search_texts ON omek_items.id = omek_search_texts.record_id
    JOIN
        omek_records_tags ON omek_items.id = omek_records_tags.record_id
    JOIN
        omek_tags ON omek_records_tags.tag_id = omek_tags.id
    JOIN
        omek_element_texts AS element_texts ON omek_items.id = element_texts.record_id AND element_texts.element_id = 40
    WHERE
        omek_items.item_type_id = 1 AND
        omek_search_texts.record_type = 'Item'
    """
    df = pd.read_sql(query, engine)

    return df


def clean_date(date_str):
    """Extract year from date string and perform sanity checks"""
    if pd.isna(date_str):
        return np.nan

    # Try to extract year from various formats
    year_match = re.search(r"(\d{4})", str(date_str))
    if year_match:
        year = int(year_match.group(1))

        # Filter to ONLY include 1901-1935
        if year < 1901 or year > 1935:
            return np.nan

        return year
    return np.nan


# Data cleaning functions
def clean_data(df):
    """Clean loaded data and print information."""
    # Check for nulls initially
    print("Initial shape:", df.shape)
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Remove rows with null values
    df = df.dropna(subset=["title", "tag_name", "date"])

    # Check for nulls after cleaning
    print("\nShape after removing null values:", df.shape)
    print("\nMissing values after removing nulls:")
    print(df.isnull().sum())

    # Extract years early and filter by range
    df["year"] = df["date"].apply(clean_date)

    # Filter to 1901-1935 range before doing anything else
    original_count = len(df)
    df = df.dropna(subset=["year"])
    print(
        f"Removed {original_count - len(df)} rows with missing or invalid years (outside 1901-1935)"
    )

    # Display sample titles
    print("\nSample of titles:")
    for i in range(min(10, len(df))):
        print(f"{i + 1}. {df.iloc[i]['title']}")

    return df


# Name extraction functions
def is_likely_person_name(name):
    """Check if a string is likely to be a person name"""
    # Names should have capital letters
    if not name or not name[0].isupper():
        return False

    # Names usually don't have more than 5 words
    if len(name.split()) > 5:
        return False

    return True


def extract_names(title):
    """Extract sender and receiver names from document titles, focusing only on real people"""
    if pd.isna(title):
        return [], []

    senders = []
    receivers = []
    title_str = str(title)

    # Pattern 1: "Person1 to Person2" - Most reliable pattern
    to_pattern = re.search(
        r"([A-Z][A-Za-z\s\.\-\']+)\s+to\s+([A-Z][A-Za-z\s\.\-\']+)[,\.]", title_str
    )
    if to_pattern:
        sender = to_pattern.group(1).strip()
        receiver = to_pattern.group(2).strip()

        # Check if these look like person names
        if is_likely_person_name(sender) and is_likely_person_name(receiver):
            senders.append(sender)
            receivers.append(receiver)
            return senders, receivers

    # Check for specific named people like "Frau Noémi P. Vetter"
    if re.search(r"Frau [\w\s\.\-]+ of Vienna", title_str):
        match = re.search(r"(Frau [\w\s\.\-]+) of Vienna", title_str)
        if match:
            person_name = match.group(1)
            senders.append(person_name)
            receivers.append("Jane Addams")
            return senders, receivers

    # Handle specific cases like "Mrs. J. T. Bowen"
    if re.search(r"Mrs\. [\w\s\.\-]+", title_str):
        match = re.search(r"(Mrs\. [\w\s\.\-]+)", title_str)
        if match:
            person_name = match.group(1)
            receivers.append(person_name)
            senders.append("Jane Addams")
            return senders, receivers

    # Add pattern for names with initials (like H. O. Hammond, J. V. Fernandez)
    initial_name_pattern = re.search(
        r"([A-Z]\.\s+[A-Z]\.\s+[A-Za-z]+|[A-Z]\.\s+[A-Z][a-z]+)\s+to\s+", title_str
    )
    if initial_name_pattern:
        sender = initial_name_pattern.group(1).strip()
        # Extract receiver if possible
        receiver_match = re.search(r"to\s+([A-Z][A-Za-z\s\.\-\']+)", title_str)
        if receiver_match:
            receiver = receiver_match.group(1).strip()
            if is_likely_person_name(receiver):
                receivers.append(receiver)

        senders.append(sender)
        return senders, receivers

    # Comprehensive list of known correspondents
    key_names = [
        "Jane Addams",
        "W. E. B. Du Bois",
        "Alice Addams",
        "Mary White Ovington",
        "Sarah Addams",
        "Theodore Parker",
        "Ellen Gates Starr",
        "Jenkin Lloyd Jones",
        "Mary Rozet Smith",
        "Emily Greene Balch",
        "Paul Underwood Kellogg",
        "Eleanor Daggett Karsten",
        "Lillian D. Wald",
        "Sarah Alice Addams Haldeman",
        "Anita McCormick Blaine",
        "Madeleine Zabriskie Doty",
        "Dorothy Detzer",
        "Mary Ryott Sheepshanks",
        "Hannah Clothier Hull",
        "Amy Woods",
        "Rosika Schwimmer",
        "Myra Harriet Reynolds Linn",
        "Florence Kelley",
        "Stanley Ross Linn",
        "Harriet Park Thomas",
        "Alice Hamilton",
        "Grace Abbott",
        "Samuel Flagg Bemis",
        "Sophonisba Breckinridge",
        "Julia Lathrop",
        "Woodrow Wilson",
        "Anna Marcet Haldeman-Julius",
        "Salmon Oliver Levinson",
        "Wilbur Kelsey Thomas",
        "Mabel L. Hyers",
        "Lucia Ames Mead",
        "Julia Clifford Lathrop",
        "Richard Theodore Ely",
        "James Grover McDonald",
        "Anna Garlin Spencer",
        "Anne Henrietta Martin",
        "Graham Taylor",
        "Mary Sheepshanks",
        "Cornelia Ramondt-Hirschmann",
        "Anna Marcet Haldeman",
        "Mina Caroline Ginger Van Winkle",
        "Louis Paul Lochner",
        "Gertrud Baer",
        "Lucy Biddle Lewis",
        "Catherine Elizabeth Marshall",
        "David Starr Jordan",
        "Carrie Chapman Catt",
        "Benjamin Barr Lindsey",
        "Abraham Isaak",
        "Theodore Roosevelt",
        "Allen B. Pond",
        "William Kent",
        "Clara Landsberg",
        "Crystal Eastman",
        "George Platt Brett Sr.",
        "William Draper Lewis",
        "Katharine Coman",
        "Marianne Beth",
        "Karl Beth",
        "Marianne Hainisch",
    ]

    # Look for mentions of specific people in the title
    found_names = []
    for name in key_names:
        if name in title_str:
            found_names.append(name)

    # If we found exactly one name, it's likely Jane Addams (the author)
    if len(found_names) == 1 and found_names[0] == "Jane Addams":
        senders.append("Jane Addams")
        return senders, receivers

    # If we found multiple names, they're likely the people involved
    if len(found_names) > 0:
        for name in found_names:
            if name != "Jane Addams":
                if "Jane Addams" in found_names:
                    # If Jane Addams is in the list, she's likely the sender
                    if "Jane Addams" not in senders:
                        senders.append("Jane Addams")
                    receivers.append(name)
                else:
                    # If Jane Addams isn't in the list, the person is likely the sender
                    senders.append(name)

        # If we have senders but no receivers, Jane Addams is likely the receiver
        if senders and not receivers and "Jane Addams" not in senders:
            receivers.append("Jane Addams")

        return senders, receivers

    # Return empty lists if no names found
    return senders, receivers


def extract_names_from_data(df):
    """Apply name extraction to dataframe."""
    senders_list = []
    receivers_list = []
    for idx, row in df.iterrows():
        s, r = extract_names(row["title"])
        senders_list.append(s)
        receivers_list.append(r)

    df["senders"] = senders_list
    df["receivers"] = receivers_list

    # Check for empty senders/receivers
    empty_names = df[
        (df["senders"].apply(lambda x: len(x) == 0))
        & (df["receivers"].apply(lambda x: len(x) == 0))
    ]
    print(f"\nNumber of items with no extracted names: {len(empty_names)}")

    # Sample these for inspection
    if len(empty_names) > 0:
        print("\nSample of titles with no extracted names:")
        for i in range(min(100, len(empty_names))):
            print(f"{i + 1}. {empty_names.iloc[i]['title']}")

    return df


# Network analysis functions
def analyze_network(df):
    """Analyze the network data and print statistics."""
    # Create a list of all people mentioned
    all_people = []
    for s in df["senders"]:
        all_people.extend(s)
    for r in df["receivers"]:
        all_people.extend(r)

    # Count frequency of each person
    people_counts = Counter(all_people)
    top_people = people_counts.most_common(50)

    # Calculate percentage of data with no extracted names
    empty_names = df[
        (df["senders"].apply(lambda x: len(x) == 0))
        & (df["receivers"].apply(lambda x: len(x) == 0))
    ]
    total_records = len(df)
    empty_records = len(empty_names)
    empty_percentage = (empty_records / total_records) * 100

    print(f"Total unique people identified: {len(people_counts)}")
    print(
        f"Records with no extracted names: {empty_records} out of {total_records} ({empty_percentage:.2f}%)"
    )

    print("\nTop 50 people mentioned in the documents:")
    for person, count in top_people:
        print(f"{person}: {count} mentions")

    # Analyze senders and receivers separately
    all_senders = []
    for s in df["senders"]:
        all_senders.extend(s)

    all_receivers = []
    for r in df["receivers"]:
        all_receivers.extend(r)

    senders_counts = Counter(all_senders)
    receivers_counts = Counter(all_receivers)

    print(f"\nUnique senders: {len(senders_counts)}")
    print(f"Unique receivers: {len(receivers_counts)}")

    # Plot distribution of top senders and receivers
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    top_senders = pd.Series(dict(senders_counts.most_common(10)))
    top_senders.plot(kind="barh")
    plt.title("Top 10 Senders")
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    top_receivers = pd.Series(dict(receivers_counts.most_common(10)))
    top_receivers.plot(kind="barh")
    plt.title("Top 10 Receivers")
    plt.tight_layout()

    plt.savefig("sender_receiver_distribution.png")
    plt.close()

    print("\nSender-Receiver distribution saved as 'sender_receiver_distribution.png'")

    return top_people


def create_network_dataset(df, top_people):
    """Create a network dataset for visualization."""
    network_data = []

    for idx, row in df.iterrows():
        senders = row["senders"]
        receivers = row["receivers"]
        tag = row["tag_name"]
        year = row["year"]

        # If we have both sender and receiver
        if senders and receivers:
            for sender in senders:
                for receiver in receivers:
                    # Avoid self-loops unless it's Jane Addams (to herself)
                    if sender != receiver or sender == "Jane Addams":
                        network_data.append(
                            {
                                "source": sender,
                                "target": receiver,
                                "tag": tag,
                                "year": year,
                                "title": row["title"],
                            }
                        )

    # Convert to DataFrame
    network_df = pd.DataFrame(network_data)

    # Print network statistics
    print("\nNetwork statistics:")
    print(f"Number of connections: {len(network_df)}")
    print(f"Number of unique sources: {network_df['source'].nunique()}")
    print(f"Number of unique targets: {network_df['target'].nunique()}")
    print(f"Number of tags in connections: {network_df['tag'].nunique()}")

    # Save the network data for visualization
    network_df.to_csv("network_data.csv", index=False)
    print("\nNetwork data saved to 'network_data.csv'")

    # Create a smaller dataset with only the main correspondents
    top_people_names = [person for person, _ in top_people[:20]]
    main_network_df = network_df[
        (network_df["source"].isin(top_people_names))
        | (network_df["target"].isin(top_people_names))
    ]

    # Save the main network data
    main_network_df.to_csv("main_network_data.csv", index=False)
    print("Main network data with top correspondents saved to 'main_network_data.csv'")

    return network_df


# Part 2: Network Preparation


def get_name_variations():
    """Return standardized name variations dictionary."""
    return {
        "Miss Addams": "Jane Addams",
        "Sarah Alice Haldeman Addams": "Sarah Alice Addams Haldeman",
        "Sarah Alice Haldeman Adams": "Sarah Alice Addams Haldeman",
        "Sarah Haldeman Addams": "Sarah Alice Addams Haldeman",
        "S. Yarros": "Rachelle Slobodinsky Yarros",
        "Anna Marcet Haldeman": "Anna Marcet Haldeman-Julius",
        "Mary Sheepshanks": "Mary Ryott Sheepshanks",
        "Jacobs and Rosa Manus": "Aletta Jacobs and Rosa Manus",
        "Anne Martin": "Anne Henrietta Martin",
        "Myra Reynolds Linn": "Myra Harriet Reynolds Linn",
        "Nina E.": "Nina E. Allender",
        "Richard T. Ely": "Richard Theodore Ely",
        "Rosika Schwimer": "Rosika Schwimmer",
        "Gandhi": "Mohandas Gandhi",
        "E.S": "Ellen Gates Starr",
    }


def get_category_colors():
    """Return category color mapping with WCAG accessible colors."""
    return {
        "Peace Work": "#2050a1",
        "Social Reform": "#0a7c5a",
        "Political Activism": "#c12f3c",
        "Personal Relations": "#6a2256",
        "Academic Work": "#a85d00",
        "General Correspondence": "#575a5c",
    }


def get_non_person_entities():
    """Return list of non-person entities to exclude."""
    return {
        "Mason-Henry Press",
        "Io Victis",
        "How Would You Uplift",
        "How Build",
        "New York Herald",
        "Trades Unions",
        "Christianity Today",
        "Charitable Effort",
        "Newer Ideas",
        "July Anticipation",
        "Macmillan Company",
        "Recent Immigration",
        "Field Neglected",
        "Other Dangers",
        "Taking Her Place",
        "New Ideals",
        "Hospital Work Among",
        "Changing Ideals",
        "Gotten Gifts",
        "Other Christian Churches",
        "Tenement Housing",
        "As Ithers See Us",
        "Chicago Federation",
        "Newsboy Conditions",
        "Newer Ideals",
        "Neighborhood Improvement",
        "Tribute",
        "American Charities",
        "Introductory Note",
        "American Street Trades",
        "Modern Philanthropy",
        "Probation Work Under Civil Service",
        "Pure Food",
        "Sinai Temple",
        "Woman Suffrage",
        "Commercial Club Dinner",
        "American Immigrants",
        "Is Class Conflict",
        "America Growing",
        "Chicago Agencies",
        "Abraham Lincoln Centre",
        "Chelsea Historical Pageant",
        "Street Trading",
        "Autobiographical Notes Upon Twenty Years",
        "Twenty Years",
        "Autobiographical Notes",
        "Unknown",
        "Ten Years",
        "Why Women Should Vote",
        "New Conscience",
        "Ancient Evil",
        "Progressive Party",
        "Modern Lear",
        "Life Above",
        "Poverty Line",
        "City Youth",
        "Administering the Funds",
        "Houghton Mifflin Company",
        "Constructive Appeal",
        "Foreign Affairs",
        "War Time",
        "Food Supply",
        "Swedish Famine",
        "Sioux City Teachers",
        "Club Women",
        "Civil ServiceAmerican Immigrant",
        "Nineteenth Century Club",
        "Anonymous",
        "More Play",
        "Factory Girls",
        "Newer Conception",
        "New World",
        "American Civil Liberties Union",
        "United States Today",
        "Koven Bowen Biography",
        "Christmas Message",
        "Community Afford",
        "Crime Unsolved",
        "Let Us Start",
        "It Anew",
        "Needed Implement",
        "Business Depressions",
        "Courageous Life",
        "Correctives Suggested",
        "End War",
        "Representative Government",
        "Republican Party Platform",
        "New Day",
        "Revealing Human Needs",
        "Christmas Day",
        "World Comity",
        "Progress Exposition",
        "Modern Woman",
        "Remarks Introducing Eleanor Roosevelt",
        "Orchestra Hall",
        "First Session",
        "Present Necessity",
        "Why Wars Must Cease",
        "Fortieth Anniversary",
        "Because Wars Interfere",
        "Normal Growth",
        "Feminist Physician Speaks",
        "E.S.",
        "Charges Against Hull",
        "The College",
        "Municipal Museum",
        "Frederick Douglass Center",
        "Municipal Museum of Chicago",
        "Civil Service",
        "More Pay",
        "Jacobs",
        "Our Moral Obligation",
        "Women Schedule",
        "City Club",
        "Mayor Turns Censor",
        "Radio Discussion With Frank Bane",
        "Birthday Poem",
        "Her Fiftieth Birthday",
        "Toynbee Hall",
        "Housing Division",
        "Character Building",
        "Hebrew Sheltering",
        "Immigrant Aid Society",
        "Old Age Security",
        "United Charities",
        "Twentieth Anniversary",
        "In Memoriam",
        "A Birthday Greeting",
        "For Jane Addams",
        "G.D.C.",
        "Illinois State Senate Bill",
        "the Chicago Institute",
        "Resolutions Committee",
        "National Conference",
        "Child Laborers",
        "The Process",
        "Relief Mobilization",
        "C.D.M",
        "The Life of Individual",
        "The Pageant",
        "Remarks on Col.",
        "Corrective Suggested",
        "Professor Freund",
        "National Education",
        "Public Words",
        "Unknown E.T",
        "Public Recreation",
        "American Immigrant",
        "Corrective Suggested",
        "American Civil Liberties",
        "Average Citizen Is Ignored",
        "Why Women Are Concerned",
        "Financial Liabilities",
        "Larger Citizenship",
        "World Politics",
        "Religious Comity",
        "Professional Women",
    }


def create_bidirectional_connection_counts(network_df):
    """Create bidirectional connection counts."""
    name_variations = get_name_variations()

    # Create pairs where the alphabetically smaller name comes first
    connection_pairs = []
    for _, row in network_df.iterrows():
        source = row["source"]
        target = row["target"]

        # Standardize name variations
        if source in name_variations:
            source = name_variations[source]
        if target in name_variations:
            target = name_variations[target]

        # Sort names alphabetically to treat A→B and B→A as the same
        if source < target:
            connection_pairs.append((source, target))
        else:
            connection_pairs.append((target, source))

    # Count the combined connections
    combined_counts = Counter(connection_pairs)

    # Create a DataFrame from the counts
    combined_df = pd.DataFrame(
        [
            {"person1": pair[0], "person2": pair[1], "total_interactions": count}
            for pair, count in combined_counts.most_common()
        ]
    )

    # Save the bidirectional connection data
    combined_df.to_csv("bidirectional_connections.csv", index=False)
    print("\nBidirectional connection data saved to 'bidirectional_connections.csv'")

    return combined_df


def get_top_connections(network_df, year, non_person_entities, name_variations):
    """Get the top connections for a specific year.

    Returns:
        tuple: (list of top people, dictionary of person counts)
    """
    # Filter for the specific year
    year_data = network_df[network_df["year"] == year]

    if len(year_data) > 0:
        # Count connections
        year_connections = []

        for _, row in year_data.iterrows():
            source = row["source"]
            target = row["target"]

            # Standardize name variations
            if source in name_variations:
                source = name_variations[source]
            if target in name_variations:
                target = name_variations[target]

            # Exclude connections involving non-person entities
            if source in non_person_entities or target in non_person_entities:
                continue

            # Skip self-connections (Jane Addams to Jane Addams)
            if source == "Jane Addams" and target == "Jane Addams":
                continue

            # Only include connections where Jane Addams is involved
            if source != "Jane Addams" and target != "Jane Addams":
                continue

            # For consistency, always put Jane Addams as the first person
            if target == "Jane Addams":
                # Swap to ensure Jane Addams is first
                source, target = target, source

            year_connections.append((source, target))

        # Count the combined connections
        year_counts = Counter(year_connections)
        top_pairs = year_counts.most_common(15)

        # Extract just the target names and their counts
        top_people = []
        person_counts = {}

        for pair, count in top_pairs:
            person = pair[1]  # The target person (not Jane Addams)
            top_people.append(person)
            person_counts[person] = count

        return top_people, person_counts

    return [], {}


def standardize_names(network_df):
    """Apply name standardization to the network data."""
    name_variations = get_name_variations()

    # Apply name standardization directly to the network_df
    for name, standard in name_variations.items():
        network_df.loc[network_df["source"] == name, "source"] = standard
        network_df.loc[network_df["target"] == name, "target"] = standard

    return network_df


def filter_jane_addams_network(network_df):
    """Filter to only include connections with Jane Addams and non-entities."""
    non_person_entities = get_non_person_entities()

    # Filter to only include connections with Jane Addams and non-entities
    addams_network = network_df[
        (
            (network_df["source"] == "Jane Addams")
            | (network_df["target"] == "Jane Addams")
        )
        & (~network_df["source"].isin(non_person_entities))
        & (~network_df["target"].isin(non_person_entities))
    ]

    # Create a column for the other person in each connection
    addams_network["other_person"] = addams_network.apply(
        lambda row: row["target"] if row["source"] == "Jane Addams" else row["source"],
        axis=1,
    )

    return addams_network


def assign_categories(addams_network):
    """Assign categories to people in the network with historical relationship context."""
    # Define category mapping based on the most common tags
    category_mapping = {
        # Social Reform category
        "Social Reform": [
            "Social",
            "Reform",
            "Settlement",
            "Hull-House",
            "Child Labor",
            "Labor",
            "Education",
        ],
        # Peace Work category
        "Peace Work": [
            "Peace",
            "War",
            "International",
            "Disarmament",
            "Arbitration",
            "Neutrality",
        ],
        # Political Activism category
        "Political Activism": [
            "Politics",
            "Woman Suffrage",
            "Progressive",
            "Democracy",
            "Government",
        ],
        # Personal Relations category
        "Personal Relations": [
            "Family",
            "Health",
            "Personal",
            "Praise",
            "Gratitude",
            "Holidays",
        ],
        # Academic Work category
        "Academic Work": [
            "Books",
            "Publishing",
            "Academic",
            "Research",
            "Lectures",
            "Writing",
        ],
    }

    # Define historical relationship weights based on Cathy's feedback
    historical_relationships = {
        # Personal Relations - family and close relationships
        "Mary Rozet Smith": {"Personal Relations": 20.0},  # Her partner
        "Sarah Alice Addams Haldeman": {"Personal Relations": 10.0},  # Her sister
        "James Weber Linn": {"Personal Relations": 10.0},  # Her nephew
        "Esther Margaret Linn Hulbert": {"Personal Relations": 10.0},  # Her niece
        # Peace Work - people known to be central to peace movements
        "Rosika Schwimmer": {"Peace Work": 10.0},
        "Lola Maverick Lloyd": {"Peace Work": 10.0},
        "Harriet Park Thomas": {"Peace Work": 10.0},
        "Gertrud Baer": {"Peace Work": 10.0},
        "Irma M. Tunas Tischer": {"Peace Work": 10.0},
        "Dorothy Detzer": {"Peace Work": 10.0},
        "Maria Matilda Widegren": {"Peace Work": 10.0},
        "Tano Jodai": {"Peace Work": 10.0},
        "Mildred Scott Olmsted": {"Peace Work": 10.0},
        "Anne Zueblin": {"Peace Work": 10.0},
        "Jeanette Rankin": {"Peace Work": 10.0},
        "Emily Greene Balch": {"Peace Work": 10.0},
        # Political Activism - political figures
        "Robert Morss Lovett": {"Political Activism": 10.0},
        "Theodore Roosevelt": {"Political Activism": 10.0},
        "Woodrow Wilson": {"Political Activism": 20.0},
        # Academic Work
        "Richard Theodore Ely": {"Academic Work": 10.0},
    }

    # Group by person and tag to find dominant themes
    person_tag_counts = (
        addams_network.groupby(["other_person", "tag"]).size().reset_index(name="count")
    )

    # Assign each person to a category based on their most common tags
    person_categories = {}

    for person, group in person_tag_counts.groupby("other_person"):
        # Get top tags for this person
        top_tags = group.sort_values("count", ascending=False)["tag"].tolist()

        # Check which category their tags match best
        category_scores = {category: 0 for category in category_mapping}

        for tag in top_tags:
            for category, keywords in category_mapping.items():
                if any(keyword.lower() in tag.lower() for keyword in keywords):
                    # Base score from tag matching
                    category_scores[category] += 1

        # Apply historical relationship weights if available
        if person in historical_relationships:
            for category, weight in historical_relationships[person].items():
                category_scores[category] += weight

        # Assign to the category with the highest score
        if any(category_scores.values()):
            top_category = max(category_scores.items(), key=lambda x: x[1])[0]
            person_categories[person] = top_category
        else:
            person_categories[person] = "General Correspondence"

    # Add category information to the network
    addams_network["category"] = addams_network["other_person"].map(person_categories)

    # Print categories for top correspondents
    print("Category assignments for top correspondents:")
    for person in addams_network["other_person"].value_counts().head(20).index:
        category = person_categories.get(person, "General Correspondence")
        print(f"{person}: {category}")

    return addams_network


def create_yearly_top_connections(addams_network):
    """Create a dataset of top connections for each year and print their categories."""
    name_variations = get_name_variations()
    non_person_entities = get_non_person_entities()

    # Analyze top connections by year
    year_range = range(
        int(addams_network["year"].min()), int(addams_network["year"].max()) + 1
    )

    print("\n===== Top 15 Connections with Jane Addams by Year =====\n")

    # Save the top 15 connections for each year in a CSV file
    top_connections_by_year = []

    for year in year_range:
        # Get top connections for this year
        top_people, person_counts = get_top_connections(
            addams_network, year, non_person_entities, name_variations
        )

        if top_people:
            # Create a dictionary to store categories for each person
            person_categories = {}

            # Get categories for each person from the network data
            for person in top_people:
                # Find this person in the data for this year
                person_data = addams_network[
                    (addams_network["other_person"] == person)
                    & (addams_network["year"] == year)
                ]

                if not person_data.empty and "category" in person_data.columns:
                    person_categories[person] = person_data["category"].iloc[0]
                else:
                    person_categories[person] = "Unknown"

            # Add data to the yearly connections dataset
            for person in top_people:
                top_connections_by_year.append(
                    {
                        "year": year,
                        "source": "Jane Addams",
                        "target": person,
                        "count": person_counts[person],
                        "category": person_categories.get(person, "Unknown"),
                    }
                )

            print(f"Year {year} - Top connections with Jane Addams:")
            for person in top_people:
                category = person_categories.get(person, "Unknown")
                print(
                    f"  Jane Addams ⟷ {person}: {person_counts[person]} interactions (Category: {category})"
                )
            print("")

    # Convert to DataFrame
    top_connections_df = pd.DataFrame(top_connections_by_year)

    # Save the top connections to a CSV file
    top_connections_df.to_csv("top_connections_by_year.csv", index=False)
    print("\nTop connections by year saved to 'top_connections_by_year.csv'.")

    return top_connections_df


def create_network_data(year, network_df):
    """Prepare network visualization data for a specific year."""
    name_variations = get_name_variations()
    non_person_entities = get_non_person_entities()
    category_colors = get_category_colors()

    # Get the top connections for this year
    top_people, person_counts = get_top_connections(
        network_df, year, non_person_entities, name_variations
    )

    if not top_people:
        return None

    # Get categories for each person
    person_categories = {}
    for person in top_people:
        # Find this person in the data
        person_data = network_df[
            (network_df["other_person"] == person) & (network_df["year"] == year)
        ]
        if not person_data.empty and "category" in person_data.columns:
            person_categories[person] = person_data["category"].iloc[0]
        else:
            person_categories[person] = "General Correspondence"

    # Calculate positions with improved layout
    positions = {}
    positions["Jane Addams"] = [0, 0]

    # Group by category
    category_people = {}
    for person in top_people:
        category = person_categories.get(person, "General Correspondence")
        if category not in category_people:
            category_people[category] = []
        category_people[category].append(person)

    # Assign angles to categories, with more spacing
    category_angles = {}
    for i, category in enumerate(category_people.keys()):
        category_angles[category] = i * (2 * np.pi / len(category_people))

    # Position people with consistent edge lengths
    uniform_distance = 1.58  # Consistent distance for all nodes from center

    # Calculate number of nodes to position
    total_nodes = sum(len(people) for people in category_people.values())

    # Ensure we have enough spacing by using fewer positions
    effective_positions = max(total_nodes, 20)  # Use at least 20 positions for spacing

    # Calculate positions in a circle with equal spacing
    for category_idx, (category, people) in enumerate(category_people.items()):
        # Get position in the overall rotation
        category_start = (
            2
            * np.pi
            * sum(
                len(cat_people)
                for cat_people in list(category_people.values())[:category_idx]
            )
            / effective_positions
        )

        # Position each person in this category
        for i, person in enumerate(people):
            # Calculate angle to ensure even spacing
            person_angle = category_start + (i * 2 * np.pi / effective_positions)

            # Avoid top-left quadrant (where categories legend is)
            if -5 * np.pi / 4 <= person_angle <= -np.pi / 4:
                # Adjust the angle to skip this quadrant
                person_angle = -np.pi / 4 + (person_angle + 3 * np.pi / 4) % (
                    2 * np.pi
                ) * (5 * np.pi / 4) / (7 * np.pi / 4)

            # Set position with uniform distance
            positions[person] = [
                uniform_distance * np.cos(person_angle),
                uniform_distance * np.sin(person_angle),
            ]

    # After positioning all nodes, check for any that are too close to center
    min_center_distance = 0.9  # Minimum acceptable distance from center
    for person in positions:
        if person != "Jane Addams":
            x, y = positions[person]
            distance = np.sqrt(x**2 + y**2)

            if distance < min_center_distance:
                # Too close to center, push outward
                angle = np.arctan2(y, x)
                new_x = uniform_distance * np.cos(angle)
                new_y = uniform_distance * np.sin(angle)
                positions[person] = [new_x, new_y]

    # Check for overlaps and adjust as needed
    overlap_iterations = 5
    for _ in range(overlap_iterations):
        overlaps_fixed = 0

        nodes = list(positions.keys())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1 :]:
                x1, y1 = positions[node1]
                x2, y2 = positions[node2]

                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if distance < 0.7 and node1 != "Jane Addams" and node2 != "Jane Addams":
                    dx, dy = x2 - x1, y2 - y1
                    if distance > 0:
                        dx, dy = dx / distance, dy / distance
                    else:
                        dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)

                    positions[node1] = [x1 - dx * 0.2, y1 - dy * 0.2]
                    positions[node2] = [x2 + dx * 0.2, y2 + dy * 0.2]
                    overlaps_fixed += 1

        if overlaps_fixed == 0:
            break

    # Create better text positions that don't overlap but stay closer to nodes
    text_positions = {}
    for person in positions:
        if person == "Jane Addams":
            text_positions[person] = "middle center"
            continue

        x, y = positions[person]
        angle = np.arctan2(y, x)
        name_length = len(person)
        is_long_name = (
            name_length > 15
        )  # Names longer than 15 chars need special handling

        if is_long_name:
            # Place long names above or below based on their position
            if y > 0:  # Upper half of the circle
                text_positions[person] = "top center"
            else:  # Lower half of the circle
                text_positions[person] = "bottom center"
        else:
            # For shorter names, use standard angle-based positioning
            if -np.pi / 8 <= angle < np.pi / 8:  # Right
                text_positions[person] = "middle right"
            elif np.pi / 8 <= angle < 3 * np.pi / 8:  # Upper right
                text_positions[person] = "top right"
            elif 3 * np.pi / 8 <= angle < 5 * np.pi / 8:  # Top
                text_positions[person] = "top center"
            elif 5 * np.pi / 8 <= angle < 7 * np.pi / 8:  # Upper left
                text_positions[person] = "top left"
            elif 7 * np.pi / 8 <= angle or angle < -7 * np.pi / 8:  # Left
                text_positions[person] = "middle left"
            elif -7 * np.pi / 8 <= angle < -5 * np.pi / 8:  # Lower left
                text_positions[person] = "bottom left"
            elif -5 * np.pi / 8 <= angle < -3 * np.pi / 8:  # Bottom
                text_positions[person] = "bottom center"
            else:  # Lower right
                text_positions[person] = "bottom right"

    # Get more details about each correspondence
    correspondence_details = {}
    for person in top_people:
        # Find this person's interactions with Jane Addams
        interactions = network_df[
            (
                (
                    (network_df["source"] == "Jane Addams")
                    & (network_df["target"] == person)
                )
                | (
                    (network_df["target"] == "Jane Addams")
                    & (network_df["source"] == person)
                )
            )
            & (network_df["year"] == year)
        ]

        # Format correspondence details with bold labels
        details = f"<b>{person}</b><br>"
        details += f"<b>Total interactions</b>: {person_counts[person]}<br>"

        # Fix category spacing by controlling the HTML precisely
        category = person_categories.get(person, "General Correspondence")
        color = category_colors.get(category, "#95a5a6")
        details += f"<b>Category</b>:<span style='color:{color};'>●</span>{category}"

        # Get all years this person appears in the data
        all_years_data = network_df[
            (
                (
                    (network_df["source"] == "Jane Addams")
                    & (network_df["target"] == person)
                )
                | (
                    (network_df["target"] == "Jane Addams")
                    & (network_df["source"] == person)
                )
            )
        ]
        active_years = sorted(all_years_data["year"].unique())

        # Show active correspondence period with bold labels
        if len(active_years) > 0:
            min_year = int(min(active_years))
            max_year = int(max(active_years))
            year_range = f"{min_year} - {max_year}"
            details += f"<br><b>Active correspondence</b>: {year_range}<br>"
            total_lifetime = len(all_years_data)
            details += f"<b>Total lifetime interactions</b>: {total_lifetime}"

        # Add tag information (topics not bold)
        if not interactions.empty:
            if "tag_name" in interactions.columns:
                tag_column = "tag_name"
            elif "tag" in interactions.columns:
                tag_column = "tag"
            else:
                tag_column = None

            if tag_column:
                tag_counts = interactions[tag_column].value_counts().to_dict()
                if tag_counts:
                    # Sort tags by frequency but don't display the counts
                    top_tags = sorted(
                        tag_counts.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    tag_names = [tag for tag, _ in top_tags]
                    details += "<br><b>Top topics</b>: " + ", ".join(tag_names)

        correspondence_details[person] = details

    # Prepare node trace
    node_x = [positions["Jane Addams"][0]]
    node_y = [positions["Jane Addams"][1]]
    node_text = ["Jane Addams"]
    node_hover = ["Jane Addams - Central Node"]
    node_sizes = [100]
    node_colors = ["rgba(140, 0, 0, 255)"]
    node_textpositions = ["middle center"]
    node_textcolors = ["white"]

    # Add other nodes
    for person in top_people:
        node_x.append(positions[person][0])
        node_y.append(positions[person][1])
        node_text.append(person)
        node_hover.append(correspondence_details[person])
        node_textcolors.append("#000000")

        # Size based on connections
        size = 20 + (person_counts[person] / max(person_counts.values())) * 30
        node_sizes.append(size)

        # Color based on category
        category = person_categories.get(person, "General Correspondence")
        node_colors.append(category_colors.get(category, "#95a5a6"))
        node_textpositions.append(text_positions[person])

    # Prepare edge traces
    edge_x = []
    edge_y = []

    for person in top_people:
        x0, y0 = positions["Jane Addams"]
        x1, y1 = positions[person]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Return all the data needed for the plot
    return {
        "node_x": node_x,
        "node_y": node_y,
        "node_text": node_text,
        "node_hover": node_hover,
        "node_sizes": node_sizes,
        "node_colors": node_colors,
        "node_textpositions": node_textpositions,
        "node_textcolors": node_textcolors,
        "edge_x": edge_x,
        "edge_y": edge_y,
        "categories_present": set(person_categories.values()),
    }


def create_interactive_visualization(data):
    """Create an interactive visualization with animation controls and left/right navigation."""
    category_colors = get_category_colors()

    # Generate data for each year
    all_years = sorted([y for y in data["year"].unique() if 1901 <= y <= 1935])
    networks_by_year = {}

    # Prepare data for each year
    for year in all_years:
        network_data = create_network_data(year, data)
        if network_data:
            networks_by_year[year] = network_data

    # Make sure we have data
    if not networks_by_year:
        print("No data available for visualization")
        return None

    # Collect ALL unique categories across all years
    all_categories = set()
    for network in networks_by_year.values():
        all_categories.update(network["categories_present"])

    # Convert to a sorted list for consistent ordering
    all_categories = sorted(list(all_categories))
    print(f"All possible categories across all years: {all_categories}")

    # Create the initial figure with data from the first available year
    first_year = min(networks_by_year.keys())
    initial_data = networks_by_year[first_year]

    # Create figure
    fig = go.Figure()

    # First add the edge trace
    fig.add_trace(
        go.Scatter(
            x=initial_data["edge_x"],
            y=initial_data["edge_y"],
            line=dict(width=0.8, color="#888888"),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
            name="Edges",
        )
    )

    # Add separate trace just for Jane Addams node and text
    fig.add_trace(
        go.Scatter(
            x=[initial_data["node_x"][0]],
            y=[initial_data["node_y"][0]],
            mode="markers+text",
            marker=dict(
                size=initial_data["node_sizes"][0],
                color=initial_data["node_colors"][0],
                line=dict(width=0),
                opacity=1,
            ),
            text=["Jane Addams"],
            textposition="middle center",
            textfont=dict(size=14, color="#f5f5f5"),
            hovertext=["Jane Addams - Central Node"],
            hoverinfo="text",
            showlegend=False,
            name="Jane Addams",
        )
    )

    # Add other nodes EXCLUDING Jane Addams
    fig.add_trace(
        go.Scatter(
            x=initial_data["node_x"][1:],
            y=initial_data["node_y"][1:],
            mode="markers+text",
            marker=dict(
                size=initial_data["node_sizes"][1:],
                color=initial_data["node_colors"][1:],
                line=dict(width=1, color="#ffffff"),
                opacity=1,
            ),
            text=initial_data["node_text"][1:],
            textposition=initial_data["node_textpositions"][1:],
            textfont=dict(size=11, color="#000000"),
            hovertext=initial_data["node_hover"][1:],
            hoverinfo="text",
            showlegend=False,
            name="Other Nodes",
        )
    )

    # Add ALL category traces, but only show those present in first year
    for category in all_categories:
        # Check if this category is present in the first year
        is_visible = category in initial_data["categories_present"]

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=category_colors.get(category, "#95a5a6")),
                name=category,
                showlegend=True,
                visible=is_visible,
            )
        )

    # Create frames for each year
    frames = []
    years_list = sorted(networks_by_year.keys())
    for year in years_list:
        network = networks_by_year[year]
        frame_data = [
            # Edge trace
            go.Scatter(
                x=network["edge_x"],
                y=network["edge_y"],
                line=dict(width=0.8, color="#888888"),
                hoverinfo="none",
                mode="lines",
                showlegend=False,
            ),
            go.Scatter(
                x=[network["node_x"][0]],
                y=[network["node_y"][0]],
                mode="markers+text",
                marker=dict(
                    size=network["node_sizes"][0],
                    color=network["node_colors"][0],
                    line=dict(width=0),
                    opacity=1,
                ),
                text=["Jane Addams"],
                textposition="middle center",
                textfont=dict(size=14, color="#f5f5f5"),
                hovertext=["Jane Addams - Central Node"],
                hoverinfo="text",
                showlegend=False,
            ),
            # Other nodes
            go.Scatter(
                x=network["node_x"][1:],
                y=network["node_y"][1:],
                mode="markers+text",
                marker=dict(
                    size=network["node_sizes"][1:],
                    color=network["node_colors"][1:],
                    line=dict(width=1, color="#ffffff"),
                    opacity=1,
                ),
                text=network["node_text"][1:],
                textposition=network["node_textpositions"][1:],
                textfont=dict(size=11, color="#000000"),
                hovertext=network["node_hover"][1:],
                hoverinfo="text",
                showlegend=False,
            ),
        ]

        # Add ALL category traces, but set visibility based on presence in this year
        for category in all_categories:
            is_visible = category in network["categories_present"]

            frame_data.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        size=10, color=category_colors.get(category, "#95a5a6")
                    ),
                    name=category,
                    showlegend=True,
                    visible=is_visible,
                )
            )

        # Create the frame with all traces
        frame = go.Frame(
            data=frame_data,
            name=str(int(year)),
        )
        frames.append(frame)

    # Add frames to figure
    fig.frames = frames

    # Add navigation controls with left/right arrows
    add_year_navigation(fig, years_list)

    return fig


def add_year_navigation(fig, years_list):
    """Add navigation controls including left and right arrow buttons to the figure."""
    # Set up the layout with title
    fig.update_layout(
        title={
            "text": "The Social World of Jane Addams (1901-1935)<br><span style='font-size:16px;'>Mapping Jane Addams' Correspondence Network Across Multiple Spheres of Influence</span>",
            "y": 0.94,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"color": "#23231b", "size": 24},
        },
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            title="Categories",
            xanchor="right",
            yanchor="top",
            x=0.99,
            y=0.99,
            bgcolor="rgba(255, 255, 255, 0.5)",
            font=dict(color="#23231b"),
        ),
        margin=dict(b=20, l=5, r=5, t=100),
        width=1000,
        height=800,
        plot_bgcolor="rgba(189, 171, 145, 0.1)",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-3.0, 3.0],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-3.0, 3.0],
        ),
    )

    # Create the slider steps with all labels visible
    slider_steps = []
    for year in years_list:
        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [str(int(year))],
                    dict(
                        frame=dict(duration=1600, redraw=True),
                        mode="immediate",
                        transition=dict(duration=1200),
                    ),
                ],
                label=str(int(year)),  # Show all year labels
            )
        )

    # Create year slider
    sliders = [
        dict(
            active=0,
            currentvalue={
                "visible": True,
                "prefix": "Year: ",
                "xanchor": "left",
            },
            steps=slider_steps,
            len=0.9,
            x=0.05,
            y=0.09,
            yanchor="top",
            bgcolor="rgba(122, 94, 55, 0.9)",
            activebgcolor="rgba(122, 94, 55, 1)",
        )
    ]

    # Create updatemenus for play & pause buttons
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=1600, redraw=True),
                            fromcurrent=True,
                            mode="immediate",
                            transition=dict(duration=1200),
                        ),
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                ),
            ],
            direction="left",
            pad=dict(r=10, t=85),
            x=0.6,
            y=0.20,
            bgcolor="rgba(189, 171, 145, 255)",
        ),
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="◀",
                    method="animate",
                    args=[
                        ["previous"],
                        dict(
                            frame=dict(duration=300, redraw=True),
                            mode="immediate",
                            transition=dict(duration=300),
                        ),
                    ],
                ),
                dict(
                    label="▶",
                    method="animate",
                    args=[
                        ["next"],
                        dict(
                            frame=dict(duration=300, redraw=True),
                            mode="immediate",
                            transition=dict(duration=300),
                        ),
                    ],
                ),
            ],
            direction="left",
            pad=dict(r=10, t=85),
            x=1.0,
            y=0.20,
            bgcolor="rgba(189, 171, 145, 255)",
        ),
    ]

    # Update layout with sliders and buttons
    fig.update_layout(
        updatemenus=updatemenus,
        sliders=sliders,
    )

    return fig


def save_interactive_visualization(fig):
    """Save the interactive visualization to HTML file with working arrow buttons."""
    # Extract years from the frames
    years_list = [int(frame.name) for frame in fig.frames]
    years_js_array = "[" + ",".join(map(str, years_list)) + "]"
    # Create additional HTML for the left/right navigation
    additional_html = f"""
    <style>
        #button-container {{
            position: absolute;
            bottom: 40px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 10px;
            z-index: 10000;
        }}

        .year-nav-button {{
            position: relative;
            background-color: rgba(74, 110, 172, 0.9);
            color: white;
            border: 3px solid white;
            border-radius: 50%;
            width: 50px;
            height: 40px;
            font-size: 20px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}

        /* Target all possible slider elements */
        .js-plotly-plot .slider-container *,
        .slider-container .slider-rect,
        .slider-container .slider-line,
        .slider-container .slider-thumb,
        [class*="slider"] {{
            cursor: pointer !important;
        }}
    </style>

    <script>
        // Wait for the page to fully load
        window.addEventListener('load', function() {{
            // Variables to track state
            let currentYearIndex = 0;
            let isPlaying = false;
            let playInterval = null;
            const years = {years_js_array};

            // Find the plotly graph container
            const plotlyContainer = document.querySelector('.plotly');
            if (!plotlyContainer) return;

            // Get the ID of the plotly graph (the parent element)
            const graphElement = plotlyContainer.closest('[id]');
            const graphId = graphElement ? graphElement.id : null;
            if (!graphId) return;

            const graphDiv = document.getElementById(graphId);

            // Function to force-stop any animations//hello
            function stopAllAnimations() {{
                isPlaying = false;
                clearInterval(playInterval);

                // Stop any Plotly animations
                if (graphDiv && graphDiv._transitioning) {{
                    Plotly.animate(graphId, [years[currentYearIndex].toString()], {{
                        mode: 'immediate',
                        transition: {{ duration: 0 }},
                        frame: {{ duration: 0 }}
                    }});
                }}
            }}

            // Call stopAllAnimations after load to prevent auto-animation
            setTimeout(stopAllAnimations, 100);

            // Function to update all UI elements to match current year
            function updateUIForYear(yearIndex) {{
                if (yearIndex < 0) yearIndex = 0;
                if (yearIndex >= years.length) yearIndex = years.length - 1;

                currentYearIndex = yearIndex;
                const year = years[currentYearIndex];

                // Animate to the selected year
                Plotly.animate(graphId, [year.toString()], {{
                    transition: {{ duration: 1200 }},
                    frame: {{ duration: 1600 }}
                }});

                // Update slider UI (this part is tricky with Plotly sliders)
                try {{
                    // Find the slider element and update it
                    const sliderEl = document.querySelector('.slider-' + currentYearIndex);
                    if (sliderEl) sliderEl.click();
                }} catch (e) {{
                    console.log("Couldn't update slider UI directly");
                }}

                // Update year display text
                const yearDisplay = document.querySelector('.slider-current-value');
                if (yearDisplay) {{
                    yearDisplay.textContent = 'Year: ' + year;
                }}
            }}

            // Play function - advances through years
            function playAnimation() {{
                isPlaying = true;
                clearInterval(playInterval); // Clear any existing interval

                playInterval = setInterval(() => {{
                    if (currentYearIndex >= years.length - 1) {{
                        // Loop back to beginning when reaching the end
                        currentYearIndex = 0;
                    }} else {{
                        currentYearIndex++;
                    }}
                    updateUIForYear(currentYearIndex);
                }}, 1000); //
            }}

            // Pause function
            function pauseAnimation() {{
                isPlaying = false;
                clearInterval(playInterval);
            }}

            // Previous year function
            function previousYear() {{
                pauseAnimation(); // Stop any running animation
                updateUIForYear(currentYearIndex - 1);
            }}

            // Next year function
            function nextYear() {{
                pauseAnimation(); // Stop any running animation
                updateUIForYear(currentYearIndex + 1);
            }}

            // Watch for Plotly button clicks
            if (graphDiv) {{
                graphDiv.on('plotly_buttonclicked', function(data) {{
                    if (data.menu && data.button) {{
                        const buttonLabel = data.button.label;

                        if (buttonLabel === '◀') {{  // Previous button
                            previousYear();
                        }} else if (buttonLabel === '▶') {{  // Next button
                            nextYear();
                        }} else if (buttonLabel === 'Play') {{
                            playAnimation();
                        }} else if (buttonLabel === 'Pause') {{
                            pauseAnimation();
                        }}
                    }}
                }});

                // Watch for slider changes
                graphDiv.on('plotly_sliderchange', function(data) {{
                    // Update current year index when slider changes
                    if (data && data.slider && typeof data.slider.active === 'number') {{
                        pauseAnimation(); // Stop any running animation
                        currentYearIndex = data.slider.active;
                    }}
                }});
            }}

            // Find the current year from the slider to initialize
            const currentYearElement = document.querySelector('.slider-current-value');
            if (currentYearElement) {{
                const yearText = currentYearElement.textContent;
                const yearMatch = yearText.match(/Year: (\\d+)/);
                if (yearMatch) {{
                    const year = parseInt(yearMatch[1]);
                    currentYearIndex = years.indexOf(year);
                    if (currentYearIndex === -1) currentYearIndex = 0;
                }}
            }}

            // Set cursor styling for slider elements
            setTimeout(() => {{
                const sliderElements = document.querySelectorAll('.slider-container .slider-rect, .slider-container .slider-line, .slider-container .slider-thumb');
                sliderElements.forEach(element => {{
                    element.style.cursor = 'pointer';
                }});

                // For the draggable handle
                const sliderThumbs = document.querySelectorAll('.slider-container .slider-thumb');
                sliderThumbs.forEach(thumb => {{
                    thumb.style.cursor = 'grab';
                }});
            }}, 1000);
        }});
    </script>
    """

    # Save the figure to an HTML file
    fig.write_html("index.html", include_plotlyjs=True, full_html=True)

    # Now open the file, add our custom HTML, and save it again
    with open("index.html", "r") as file:
        html_content = file.read()

    # Insert our custom HTML before the </body> tag
    updated_html = html_content.replace("</body>", additional_html + "</body>")

    with open("index.html", "w") as file:
        file.write(updated_html)

    print("Interactive visualization with arrow navigation saved to 'index.html'")


# Complete main function integrating all parts
def main():
    """Main function integrating all parts of the analysis and visualization."""
    # === Part 1: Data Processing ===
    print("\n=== Starting Data Processing (Part 1) ===\n")

    # Load data
    print("Loading data...")
    df = load_data()

    # Clean data and filter by year range (1901-1935) early
    print("Cleaning data and filtering to 1901-1935...")
    df = clean_data(df)

    # Extract names
    print("Extracting names for 1901-1935 data...")
    df = extract_names_from_data(df)

    # Analyze network
    print("Analyzing network...")
    top_people = analyze_network(df)

    # Create network dataset
    print("Creating network dataset...")
    network_df = create_network_dataset(df, top_people)

    # === Part 2: Network Preparation ===
    print("\n=== Starting Network Preparation (Part 2) ===\n")

    # Standardize names
    print("Standardizing names...")
    network_df = standardize_names(network_df)

    # Create bidirectional connections
    print("Creating bidirectional connection counts...")
    bidirectional_df = create_bidirectional_connection_counts(network_df)

    # Filter to Jane Addams network
    print("Filtering to Jane Addams network...")
    addams_network = filter_jane_addams_network(network_df)

    # Assign categories
    print("Assigning categories to correspondents...")
    addams_network = assign_categories(addams_network)

    # Create yearly top connections
    print("Creating yearly top connections...")
    top_connections_df = create_yearly_top_connections(addams_network)

    # Save the categorized network for visualization
    addams_network.to_csv("jane_addams_categorized_network.csv", index=False)
    print("Categorized network saved to 'jane_addams_categorized_network.csv'")

    # === Part 3: Visualization ===
    print("\n=== Starting Visualization Generation (Part 3) ===\n")

    # Create interactive visualization
    print("Creating interactive visualization...")
    interactive_fig = create_interactive_visualization(addams_network)

    if interactive_fig:
        # Save the interactive visualization with navigation arrows
        print("Saving interactive visualization...")
        save_interactive_visualization(interactive_fig)
        print("Interactive visualization saved successfully.")
    else:
        print("Failed to create interactive visualization.")

        print("\nAll analysis and visualization complete!")

    return df, network_df, addams_network, interactive_fig


if __name__ == "__main__":
    main()
