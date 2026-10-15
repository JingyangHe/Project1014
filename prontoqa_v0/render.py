"""Controlled English renderer and strict inverse parser; no logical string edits."""
import hashlib
import json
import re
from .logic import Fact, Literal, Rule, World, Rejected

INSTRUCTION = "Use only the stated facts and rules. Answer the query with True or False."


def category_phrase(predicate):
    return ("an " if predicate[0] in "aeiou" else "a ") + predicate


def canonical_input_json(question: str, query: str) -> str:
    """Exact text identity: no whitespace, case, name or Unicode normalization."""
    if not isinstance(question, str) or not isinstance(query, str):
        raise TypeError("question and query must be strings")
    return json.dumps({"question": question, "query": query}, ensure_ascii=False,
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def compute_example_id(question: str, query: str) -> str:
    return hashlib.sha256(canonical_input_json(question, query).encode("utf-8")).hexdigest()


def build_prompt(record) -> str:
    """Whitelist only question/query; task instruction and output format stay fixed.

    IDs, labels, proofs and all other backend metadata are intentionally unread.
    This builds the original task prompt, without any OPD student prefix.
    """
    question, query = record["question"], record["query"]
    canonical_input_json(question, query)  # Check both public fields are strings.
    return f"{INSTRUCTION}\n\n{question}\n{query}"


def build_model_input(question: str, query: str) -> str:
    """Backward-compatible two-string entry point for the same prompt builder."""
    return build_prompt({"question": question, "query": query})


def complement(literal, categories):
    if literal.predicate in categories:
        assert not literal.negative
        return category_phrase(literal.predicate)
    return ("not " if literal.negative else "") + literal.predicate


def fact_text(world, literal):
    return f"{world.entity} is {complement(literal, world.categories)}."


def rule_text(world, rule, style):
    tail = complement(rule.consequent, world.categories)
    if style == 0:
        return f"Every {rule.antecedent} is {tail}."
    if style == 1:
        return f"If someone is {category_phrase(rule.antecedent)}, then they are {tail}."
    if style == 2:
        return f"Each {rule.antecedent} is {tail}."
    raise ValueError("unknown rendering style")


def render(world, plan, proof):
    sentences = {f.id: fact_text(world, f.literal) for f in world.facts}
    sentences.update({r.id: rule_text(world, r, plan["styles"][r.id]) for r in world.rules})
    question = " ".join(sentences[ref] for ref in plan["order"])
    query = "True or false: " + fact_text(world, world.query)
    rendered_proof = []
    for step in proof:
        lit = Literal(step["conclusion"]["predicate"], step["conclusion"]["negative"])
        rendered_proof.append({**step, "text": fact_text(world, lit)})
    return question, query, rendered_proof


def parse_text(question, query):
    """Parse from text alone: category vs property is marked by 'a'."""
    cats, props, facts, rules, entities = set(), set(), [], [], set()

    def parse_complement(text):
        if text.startswith(("a ", "an ")):
            predicate, negative = text.split(" ", 1)[1], False
            cats.add(predicate)
        else:
            negative = text.startswith("not ")
            predicate = text[4:] if negative else text
            props.add(predicate)
        if not re.fullmatch(r"[a-z]+", predicate):
            raise Rejected("text_parse_predicate", text)
        return Literal(predicate, negative)

    if not question.endswith("."):
        raise Rejected("text_parse_punctuation")
    for sentence in question[:-1].split(". "):
        match = re.fullmatch(r"(?:Every|Each) ([a-z]+) is (.+)", sentence)
        if not match:
            match = re.fullmatch(r"If someone is an? ([a-z]+), then they are (.+)", sentence)
        if match:
            antecedent, tail = match.groups()
            cats.add(antecedent)
            rules.append(Rule(f"r{len(rules)+1:03d}", antecedent, parse_complement(tail)))
        else:
            match = re.fullmatch(r"([A-Z][a-z]+) is (.+)", sentence)
            if not match:
                raise Rejected("text_parse_sentence", sentence)
            entity, tail = match.groups()
            entities.add(entity)
            facts.append(Fact(f"f{len(facts)+1:03d}", parse_complement(tail)))
    match = re.fullmatch(r"True or false: ([A-Z][a-z]+) is ([a-z]+)\.", query)
    if not match:
        raise Rejected("text_parse_query", query)
    entity, target = match.groups()
    entities.add(entity)
    props.add(target)
    if len(entities) != 1:
        raise Rejected("multiple_entities")
    return World(entity, tuple(sorted(cats)), tuple(sorted(props)), tuple(facts), tuple(rules), Literal(target)).check()
