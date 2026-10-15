"""Typed single-entity unary logic and shortest forward proofs."""
from dataclasses import dataclass, replace


class Rejected(ValueError):
    def __init__(self, reason, detail=""):
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True, order=True)
class Literal:
    predicate: str
    negative: bool = False

    def opposite(self):
        return replace(self, negative=not self.negative)

    def to_dict(self):
        return {"predicate": self.predicate, "negative": self.negative}


@dataclass(frozen=True)
class Fact:
    id: str
    literal: Literal


@dataclass(frozen=True)
class Rule:
    id: str
    antecedent: str
    consequent: Literal

    def to_dict(self):
        return {"id": self.id, "antecedent": self.antecedent,
                "consequent": self.consequent.to_dict()}


@dataclass(frozen=True)
class World:
    entity: str
    categories: tuple[str, ...]
    properties: tuple[str, ...]
    facts: tuple[Fact, ...]
    rules: tuple[Rule, ...]
    query: Literal

    def check(self):
        cats, props = set(self.categories), set(self.properties)
        if cats & props or len(cats) != len(self.categories) or len(props) != len(self.properties):
            raise Rejected("invalid_predicate_inventory")
        if self.query.negative or self.query.predicate not in props:
            raise Rejected("unsupported_query")
        if len({r.id for r in self.rules}) != len(self.rules) or len({f.id for f in self.facts}) != len(self.facts):
            raise Rejected("duplicate_reference_id")
        for lit in [f.literal for f in self.facts] + [r.consequent for r in self.rules]:
            if lit.predicate not in cats | props or (lit.negative and lit.predicate in cats):
                raise Rejected("unsupported_literal", str(lit))
        if any(r.antecedent not in cats for r in self.rules):
            raise Rejected("unsupported_antecedent")
        return self

    def to_dict(self):
        return {"entity": self.entity, "categories": list(self.categories),
                "properties": list(self.properties),
                "facts": [{"id": f.id, "entity": self.entity, **f.literal.to_dict()} for f in self.facts],
                "rules": [r.to_dict() for r in self.rules],
                "query": {"entity": self.entity, **self.query.to_dict()}}

    @classmethod
    def from_dict(cls, d):
        entity = d["entity"]
        if any(f["entity"] != entity for f in d["facts"]) or d["query"]["entity"] != entity:
            raise Rejected("multiple_entities")
        lit = lambda x: Literal(x["predicate"], x["negative"])
        return cls(entity, tuple(d["categories"]), tuple(d["properties"]),
                   tuple(Fact(f["id"], lit(f)) for f in d["facts"]),
                   tuple(Rule(r["id"], r["antecedent"], lit(r["consequent"])) for r in d["rules"]),
                   lit(d["query"])).check()


def solve(world):
    """Relax distances to a fixed point; facts cost zero, each rule costs one.

    No contraposition, closed-world assumption, or explosion is used.
    """
    world.check()
    distances = {f.literal: 0 for f in world.facts}
    parents = {f.literal: ("fact", f.id) for f in world.facts}
    changed = True
    while changed:
        changed = False
        for rule in world.rules:
            src = Literal(rule.antecedent)
            if src in distances and distances[src] + 1 < distances.get(rule.consequent, float("inf")):
                distances[rule.consequent] = distances[src] + 1
                parents[rule.consequent] = ("rule", rule.id)
                changed = True
    conflicts = sorted({lit.predicate for lit in distances if lit.opposite() in distances})
    yes, no = world.query in distances, world.query.opposite() in distances
    status = "contradiction" if conflicts else "True" if yes else "False" if no else "unknown"
    goal = world.query if yes else world.query.opposite()
    return {"answer": status, "hops": distances.get(goal), "distances": distances,
            "parents": parents, "conflicts": conflicts}


def require_decidable(world, hops=None):
    result = solve(world)
    if result["answer"] not in ("True", "False"):
        raise Rejected(result["answer"])
    if hops is not None and result["hops"] != hops:
        raise Rejected("wrong_shortest_hops", f"expected {hops}, got {result['hops']}")
    return result


def shortest_proof(world):
    result = require_decidable(world)
    goal = world.query if result["answer"] == "True" else world.query.opposite()
    rules = {r.id: r for r in world.rules}
    path = []
    current = goal
    while result["parents"][current][0] == "rule":
        rule_id = result["parents"][current][1]
        path.append((rule_id, current))
        current = Literal(rules[rule_id].antecedent)
    steps = [{"id": "p0", "kind": "fact", "fact_id": result["parents"][current][1],
              "conclusion": {"entity": world.entity, **current.to_dict()}}]
    for rule_id, conclusion in reversed(path):
        n = len(steps)
        steps.append({"id": f"p{n}", "kind": "rule", "rule_id": rule_id,
                      "premise": f"p{n-1}", "conclusion": {"entity": world.entity, **conclusion.to_dict()}})
    return steps


def verify_proof(world, steps, hops):
    """Check references, instantiations, goal and shortestness independently of stored text."""
    facts, rules = {f.id: f.literal for f in world.facts}, {r.id: r for r in world.rules}
    known = {}
    rule_count = 0
    for i, step in enumerate(steps):
        assert step["id"] == f"p{i}" and step["conclusion"]["entity"] == world.entity
        conclusion = Literal(step["conclusion"]["predicate"], step["conclusion"]["negative"])
        if step["kind"] == "fact":
            assert i == 0 and facts[step["fact_id"]] == conclusion
        else:
            assert step["kind"] == "rule"
            rule = rules[step["rule_id"]]
            assert known[step["premise"]] == Literal(rule.antecedent)
            assert conclusion == rule.consequent
            rule_count += 1
        known[step["id"]] = conclusion
    result = require_decidable(world, hops)
    expected = world.query if result["answer"] == "True" else world.query.opposite()
    assert steps and known[steps[-1]["id"]] == expected and rule_count == hops
    return True
