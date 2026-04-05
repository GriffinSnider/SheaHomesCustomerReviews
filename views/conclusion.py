import streamlit as st

from utils.config import SATISFIED_MIN_STARS
from utils.components import section_header, explain, commentary, finding, nav_buttons


def render(df, fdf, page):
    st.title("Conclusion")
    st.markdown("---")
    explain(
        f"This project analyzed {len(fdf):,} verified customer reviews of Shea Homes using a multi-method NLP framework spanning summary statistics, sentiment analysis, topic modeling, and predictive machine learning. The goal was to demonstrate how unstructured customer feedback can be transformed into structured, actionable intelligence at scale. Below is a summary of the key findings from each stage of the analysis, followed by strategic recommendations.")

    # key findings
    section_header("Key Findings", "What the data revealed across five stages of analysis")

    _c_pct_high = (fdf["total_score"] >= SATISFIED_MIN_STARS).mean()
    _c_at_risk_n = (fdf["total_score"] < SATISFIED_MIN_STARS).sum()
    _c_at_risk_pct = (fdf["total_score"] < SATISFIED_MIN_STARS).mean()
    finding(
        f"<b>1. Overall Customer Satisfaction Is Strong but Not Uniform</b><br>Shea Homes holds an average rating of {fdf['total_score'].mean():.2f} out of 5, with {_c_pct_high:.0%} of reviews at 4 or 5 stars. However, {_c_at_risk_n:,} reviews ({_c_at_risk_pct:.0%}) fall in the 1–3 star range, representing a meaningful segment of at-risk customers. All five rating dimensions (Overall, Quality, Trust, Value, Responsiveness) are highly correlated, meaning a poor experience in one area tends to drag down perceptions across the board.")

    _c_vader_pos = (fdf["vader_label"] == "Positive").mean()
    _c_vader_neg = (fdf["vader_label"] == "Negative").mean()
    _c_vader_neu = (fdf["vader_label"] == "Neutral").mean()
    finding(
        f"<b>2. Sentiment Analysis Confirms and Extends Star Ratings</b><br>VADER sentiment analysis classified {_c_vader_pos:.0%} of reviews as positive, {_c_vader_neg:.0%} as negative, and {_c_vader_neu:.0%} as neutral. Importantly, sentiment scores revealed nuance that star ratings alone cannot capture. Some 4- and 5-star reviews contained negative language about specific issues, while a few low-rated reviews still acknowledged positive aspects of the experience. This means relying solely on star ratings can miss customers who are satisfied overall but frustrated with specific touchpoints.")

    finding(
        "<b>3. Recurring Pain Points Are Specific and Actionable</b><br>Topic modeling and keyword analysis consistently surfaced the same friction areas: warranty responsiveness and post-close follow-through, construction quality concerns (paint, drywall, appliances, workmanship), and communication gaps during the building process. Negative reviews used concrete, specific language ('repairs,' 'months,' 'workmanship,' 'appliances'), while positive reviews used emotional, relational language ('excellent,' 'professional,' 'helpful,' 'team'). The distinction suggests that negative experiences are driven by tangible, fixable process failures rather than abstract dissatisfaction.")

    finding(
        "<b>4. Machine Learning Can Predict Satisfaction from Text Alone</b><br>Binary and three-class classification models were trained to predict customer satisfaction from review text alone. The models demonstrate that predictive text classification is viable for customer feedback that arrives without a numerical rating, such as emails, survey comments, or social media posts. Neutral reviews remain the most difficult category to classify due to their mixed-signal language. Full model performance details are available in Part 5.")

    finding(
        "<b>5. The Most Informative Signals Come from Specific Language</b><br>Across every method, from TF-IDF to logistic regression coefficients to topic models, the most predictive and diagnostic signals came from domain-specific vocabulary. Words like 'warranty,' 'repairs,' 'workmanship,' and 'months' consistently flagged dissatisfaction, while 'professional,' 'team,' 'helpful,' and 'awesome' signaled satisfaction. This confirms that targeted keyword monitoring, even without full ML pipelines, can provide early warning of emerging issues.")

    # strategic recommendations
    section_header("Strategic Recommendations", "How these findings could translate into action")

    commentary(
        "<b>1. Implement a Real-Time Review Monitoring Pipeline</b><br>The NLP methods demonstrated in this project, specifically VADER sentiment scoring and keyword flagging, could be deployed as an automated pipeline that scores each new review as it arrives. Reviews falling below a sentiment threshold or containing high-risk keywords (e.g., 'warranty,' 'months,' 'repairs') could be automatically routed to the appropriate regional team for follow-up. This would reduce the lag between a customer expressing dissatisfaction and the company responding to it.")

    commentary(
        "<b>2. Prioritize Warranty and Post-Close Communication</b><br>The most consistent theme in negative reviews across every analytical method was frustration with warranty responsiveness and post-close follow-through. Customers who rated construction quality poorly often cited delays in getting issues resolved after move-in, not necessarily the defects themselves. Strengthening the warranty response process and setting clear timelines for issue resolution could meaningfully shift perception in this segment.")

    commentary(
        "<b>3. Use Predictive Models to Flag Unrated Feedback</b><br>The machine learning classifiers trained in this project could be applied to customer feedback channels that do not include star ratings, such as warranty service tickets, post-closing survey comments, or email correspondence. By scoring this text-only feedback with the same models, Shea Homes could identify at-risk customers earlier in the lifecycle, before they reach the point of writing a negative public review.")

    commentary(
        "<b>4. Track Sentiment Trends at the Market and Community Level</b><br>Aggregated national ratings can mask localized issues. The geographic analysis showed variation in satisfaction across states and cities. Running sentiment analysis at the community or project level on a recurring basis would allow regional managers to identify emerging issues before they become systemic patterns, and would provide data-driven evidence for resource allocation decisions.")

    # closing
    section_header("Final Note")

    explain(
        "This project was built as a portfolio demonstration of applied NLP and machine learning using real-world customer data. The full analytical pipeline, from data collection through web scraping to predictive modeling, was developed independently by Griffin Snider.<br><br>The techniques shown here are not theoretical. Every method used in this project, from VADER sentiment scoring to TF-IDF topic extraction to Gradient Boosting classification, can be deployed on live data in a production environment. The goal was to show what becomes possible when customer feedback is treated not as a static archive, but as a continuous, analyzable data source.<br><br>Thank you for reviewing this project.")

    nav_buttons(page)
