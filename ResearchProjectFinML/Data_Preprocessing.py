import pandas as pd
from edgar import Company

# Analyze based on Business, Risk Factors, and Management's Discussion and Analysis of Financial Condition and Results of Operations

# Define the company ticker
ticker = "AAPL"

# Define the range of years for which you want to retrieve 10-K filings
start_year = 1995
end_year = 2023

# Initialize an empty DataFrame to store all items
all_items_df = pd.DataFrame(columns=["Item", "Content"])

# Retrieve 10-K filings for the specified range of years
filings = Company(ticker).get_filings(form="10-K")
filings.filter(date=f"{start_year}-01-01:{end_year}-12-31")

# Iterate over each filing for the current year
for filing in filings:
    # Retrieve the HTML content of the filing
    html = filing.html()
    
    if html:
        # Create a ChunkedDocument object from the HTML content
        chunked_document = ChunkedDocument(html)
    
        # Extract items (sections) from the filing based on a condition
        items = chunked_document.show_items("Item.str.contains('ITEM', case=False)", "Item")
        
        # Initialize a dictionary to store content for each item
        item_contents = {}
        
        # Iterate over each item and extract content
        for index, row in items.iterrows():
            # Store the content in reverse order to correct the concatenation
            item_contents[row["Item"]] = item_contents.get(row["Item"], "") + row["Text"]
        
        # Create a DataFrame from the item contents
        filing_df = pd.DataFrame.from_dict(item_contents, orient="index", columns=["Content"])
        filing_df.reset_index(inplace=True)
        filing_df.columns = ["Item", "Content"]
        
        # Concatenate the extracted items to the DataFrame containing all items
        all_items_df = pd.concat([all_items_df, filing_df], ignore_index=True)

# Print the DataFrame containing all items
# Define the list of valid items
valid_items = ["Item 1", "Item 1A", "Item 7"]

# Filter the DataFrame to keep only the rows with valid items
filtered_df = all_items_df[all_items_df['Item'].isin(valid_items)]

# Print the filtered DataFrame
print(filtered_df)