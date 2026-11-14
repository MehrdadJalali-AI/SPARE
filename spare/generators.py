"""Synthetic benchmark generators.

ub_synth   LUBM-style generator following the Univ-Bench schema (own
           re-implementation, not the original Java UBA). Two TBox profiles:
           'rdfs' (linear rules only) and 'rl' (adds transitivity, a property
           chain, existential and conjunctive left-hand sides).
synth_j    random TBox/ABox with a knob p_multi controlling the share of
           multi-premise axioms, used to probe the complexity dichotomy.
"""
from __future__ import annotations

import random
from typing import List, Tuple

from .kb import TYPE, TBox, Fact

UB = "ub:"


def ub_tbox(profile: str = "rl") -> TBox:
    u = lambda x: UB + x
    sc = [("FullProfessor", "Professor"), ("AssociateProfessor", "Professor"),
          ("AssistantProfessor", "Professor"), ("Professor", "Faculty"),
          ("Lecturer", "Faculty"), ("Faculty", "Employee"), ("Employee", "Person"),
          ("GraduateStudent", "Student"), ("UndergraduateStudent", "Student"),
          ("Student", "Person"), ("University", "Organization"),
          ("Department", "Organization"), ("ResearchGroup", "Organization"),
          ("GraduateCourse", "Course"), ("Course", "Work"),
          ("JournalArticle", "Article"), ("ConferencePaper", "Article"),
          ("Article", "Publication"), ("Chair", "Person"),
          ("TeachingAssistant", "Person"), ("ResearchAssistant", "Person"),
          ("WorkingStudent", "Person"), ("GraduateTA", "TeachingAssistant")]
    sp = [("headOf", "worksFor"), ("worksFor", "memberOf"),
          ("undergraduateDegreeFrom", "degreeFrom"), ("mastersDegreeFrom", "degreeFrom"),
          ("doctoralDegreeFrom", "degreeFrom")]
    dom = [("memberOf", "Person"), ("teacherOf", "Faculty"), ("advisor", "Person"),
           ("publicationAuthor", "Publication"), ("degreeFrom", "Person"),
           ("subOrganizationOf", "Organization"), ("teachingAssistantOf", "TeachingAssistant")]
    rng = [("memberOf", "Organization"), ("teacherOf", "Course"), ("takesCourse", "Course"),
           ("advisor", "Professor"), ("publicationAuthor", "Person"),
           ("degreeFrom", "University"), ("subOrganizationOf", "Organization"),
           ("teachingAssistantOf", "Course")]
    inv = [("member", "memberOf"), ("hasAlumnus", "degreeFrom")]
    t = TBox(subclass=[(u(a), u(b)) for a, b in sc],
             subproperty=[(u(a), u(b)) for a, b in sp],
             domain=[(u(a), u(b)) for a, b in dom],
             range=[(u(a), u(b)) for a, b in rng],
             inverse=[(u(a), u(b)) for a, b in inv])
    if profile == "rdfs":
        return t
    t.transitive = [u("subOrganizationOf")]
    t.chain = [(u("memberOf"), u("subOrganizationOf"), u("memberOf"))]
    t.exists = [(u("headOf"), u("Department"), u("Chair")),
                (u("worksFor"), u("ResearchGroup"), u("ResearchAssistant")),
                (u("takesCourse"), u("GraduateCourse"), u("GraduateStudentRole")),
                (u("advisor"), u("FullProfessor"), u("SeniorSupervised"))]
    t.conj = [(u("Student"), u("Employee"), u("WorkingStudent")),
              (u("GraduateStudent"), u("TeachingAssistant"), u("GraduateTA")),
              (u("Professor"), u("Chair"), u("ProfessorChair"))]
    return t


def ub_synth(n_univ: int = 1, n_dept: int = 2, scale: float = 0.3, seed: int = 0,
             profile: str = "rl") -> Tuple[TBox, List[Fact]]:
    rng = random.Random(seed)
    A: List[Fact] = []
    T = lambda x, c: A.append((x, TYPE, UB + c))
    R = lambda s, p, o: A.append((s, UB + p, o))
    k = lambda lo, hi: max(1, round(rng.randint(lo, hi) * scale)) if scale < 1 else rng.randint(lo, hi)
    unis = [f"u{i}" for i in range(max(n_univ, 3))]
    for u in unis:
        T(u, "University")
    for ui in range(n_univ):
        u = unis[ui]
        for di in range(n_dept):
            d = f"{u}.d{di}"
            T(d, "Department")
            R(d, "subOrganizationOf", u)
            groups = [f"{d}.g{i}" for i in range(k(2, 5))]
            for g in groups:
                T(g, "ResearchGroup")
                R(g, "subOrganizationOf", d)
            courses = [f"{d}.c{i}" for i in range(k(10, 16))]
            gcourses = [f"{d}.gc{i}" for i in range(k(6, 10))]
            for c in courses:
                T(c, "Course")
            for c in gcourses:
                T(c, "GraduateCourse")
            fac = []
            for kind, lo, hi in [("FullProfessor", 4, 7), ("AssociateProfessor", 5, 8),
                                 ("AssistantProfessor", 4, 6), ("Lecturer", 2, 4)]:
                for i in range(k(lo, hi)):
                    f = f"{d}.{kind[:2]}{i}"
                    fac.append((f, kind))
                    T(f, kind)
                    R(f, "worksFor", d)
                    for c in rng.sample(courses + gcourses, k=min(2, len(courses) + len(gcourses))):
                        R(f, "teacherOf", c)
                    R(f, "undergraduateDegreeFrom", rng.choice(unis))
                    R(f, "doctoralDegreeFrom", rng.choice(unis))
                    if rng.random() < 0.3:
                        R(f, "worksFor", rng.choice(groups))
            head = next(f for f, kd in fac if kd == "FullProfessor")
            R(head, "headOf", d)
            profs = [f for f, kd in fac if kd != "Lecturer"]
            fulls = [f for f, kd in fac if kd == "FullProfessor"]
            for i in range(k(25, 40)):
                s = f"{d}.gs{i}"
                T(s, "GraduateStudent")
                R(s, "memberOf", d)
                for c in rng.sample(gcourses, k=min(len(gcourses), rng.randint(1, 3))):
                    R(s, "takesCourse", c)
                R(s, "advisor", rng.choice(profs if rng.random() < 0.7 else fulls))
                R(s, "undergraduateDegreeFrom", rng.choice(unis))
                if rng.random() < 0.25:
                    R(s, "teachingAssistantOf", rng.choice(courses))
                if rng.random() < 0.25:
                    R(s, "worksFor", rng.choice(groups))
            for i in range(k(60, 90)):
                s = f"{d}.us{i}"
                T(s, "UndergraduateStudent")
                R(s, "memberOf", d)
                for c in rng.sample(courses, k=min(len(courses), rng.randint(2, 4))):
                    R(s, "takesCourse", c)
                if rng.random() < 0.2:
                    R(s, "advisor", rng.choice(profs))
            for f, kd in fac:
                for i in range(rng.randint(1, 4)):
                    p = f"{f}.pub{i}"
                    T(p, rng.choice(["JournalArticle", "ConferencePaper"]))
                    R(p, "publicationAuthor", f)
    return ub_tbox(profile), sorted(set(A))


def synth_j(p_multi: float = 0.5, n_classes: int = 40, n_props: int = 12,
            n_ind: int = 150, n_edges: int = 300, seed: int = 0) -> Tuple[TBox, List[Fact]]:
    """Random KG whose TBox has a controllable share of multi-premise axioms."""
    rng = random.Random(seed)
    C = [f"C{i}" for i in range(n_classes)]
    P = [f"p{i}" for i in range(n_props)]
    t = TBox()
    for i in range(1, n_classes):
        if rng.random() < 0.7:
            t.subclass.append((C[i], C[rng.randrange(0, i)]))
    for p in P:
        if rng.random() < 0.5:
            t.domain.append((p, rng.choice(C)))
        if rng.random() < 0.5:
            t.range.append((p, rng.choice(C)))
    n_lin = len(t.subclass) + len(t.domain) + len(t.range)
    n_multi = round(p_multi * n_lin / max(1e-9, (1 - p_multi))) if p_multi < 1 else n_lin
    n_multi = min(n_multi, 3 * n_lin)
    for _ in range(n_multi):
        r = rng.random()
        if r < 0.5:
            a, b = rng.sample(C, 2)
            t.conj.append((a, b, rng.choice(C)))
        elif r < 0.9:
            t.exists.append((rng.choice(P), rng.choice(C), rng.choice(C)))
        else:
            p1, p2 = rng.sample(P, 2)
            t.chain.append((p1, p2, rng.choice(P)))
    if p_multi >= 1:
        t.subclass, t.domain, t.range = [], [], []
    I = [f"i{k}" for k in range(n_ind)]
    A = []
    for x in I:
        for c in rng.sample(C, rng.randint(1, 3)):
            A.append((x, TYPE, c))
    for _ in range(n_edges):
        A.append((rng.choice(I), rng.choice(P), rng.choice(I)))
    return t, sorted(set(A))


def dks_instance(n: int = 12, p: float = 0.3, seed: int = 0):
    """The Theorem-1 reduction: graph G -> BJPS instance (one individual x,
    P_u(x) per vertex, axiom P_u ⊓ P_v ⊑ E_uv per edge)."""
    rng = random.Random(seed)
    t = TBox()
    A = [("x", TYPE, f"P{u}") for u in range(n)]
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    for u, v in edges:
        t.conj.append((f"P{u}", f"P{v}", f"E{u}_{v}"))
    return t, A, edges
