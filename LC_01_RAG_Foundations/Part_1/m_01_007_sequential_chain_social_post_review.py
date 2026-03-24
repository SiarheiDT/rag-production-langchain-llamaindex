"""
Example 7: Sequential LCEL chain.

This script reproduces the lesson example where:
1) the model generates a social media post,
2) another prompt reviews that post.

Usage:
    export OPENAI_API_KEY=...
    python m_01_007_sequential_chain_social_post_review.py
"""

from common.env_loader import load_env
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


def main() -> None:
    load_env()

    post_prompt = PromptTemplate.from_template(
        """You are a business owner. Given the theme of a post, write a social media post to share on my socials.

Theme: {theme}
Content: This is social media post based on the theme above:"""
    )

    review_prompt = PromptTemplate.from_template(
        """You are an expert social media manager. Given the presented social media post, it is your job to write a review for the post.

Social Media Post:
{post}
Review from a Social Media Expert:"""
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    chain = (
        {"post": post_prompt | llm | StrOutputParser()}
        | review_prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke({"theme": "Having a black friday sale with 50% off on everything."})
    print(result)


if __name__ == "__main__":
    main()
