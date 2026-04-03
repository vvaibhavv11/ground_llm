from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_EXPORTS = ("build_info", "encode_train", "encode", "decode_string")


def ensure_project_site_packages() -> None:
    matches = sorted((ROOT / ".venv" / "lib").glob("python*/site-packages"))
    if not matches:
        return

    site_packages = str(matches[0])
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)


def load_ground_llm():
    ensure_project_site_packages()
    ground_llm = importlib.import_module("ground_llm")
    if all(hasattr(ground_llm, name) for name in EXPECTED_EXPORTS):
        return ground_llm

    subprocess.run(
        [sys.executable, "-m", "maturin", "develop"],
        cwd=ROOT,
        check=True,
    )
    ensure_project_site_packages()
    importlib.invalidate_caches()
    sys.modules.pop("ground_llm.ground_llm", None)
    sys.modules.pop("ground_llm", None)
    ground_llm = importlib.import_module("ground_llm")
    if all(hasattr(ground_llm, name) for name in EXPECTED_EXPORTS):
        return ground_llm

    raise RuntimeError(
        "ground_llm is missing expected Rust bindings after `maturin develop`"
    )


def summarize_text(text: str, preview: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= preview:
        return compact
    return f"{compact[:preview]}..."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test the Rust tokenizer bindings"
    )
    parser.add_argument(
        "--text",
        default="banana bandana",
        help="Input text used for training/encoding/decoding",
    )
    args = parser.parse_args()
    ground_llm = load_ground_llm()

    text = """The quiet power of a morning walk can reset your internal clock.
    A brisk walk before sunrise can reset your internal clock.
    The world is still, and the air carries a crisp, almost metallic scent.
    Your mind, unburdened by the day's demands, begins to sift through thoughts.
    Notice the way the sun paints the sky in hues of amber and lavender.
    Each step is a gentle reminder that movement nurtures both body and soul.
    When you return home, the fatigue of the night feels distant.
    You’re ready to face the day with clarity and calm.

    The art of listening in digital communication.
    Typing a reply often feels like a race against time.
    Yet, genuine listening—reading between the lines—creates deeper connections.
    Pause before you respond; consider the sender’s tone and intent.
    A simple “I hear you” can transform a text thread into a dialogue.
    Remember, emojis are tools, not replacements for empathy.
    When you ask follow‑up questions, you show that you care.
    Digital conversations thrive when we prioritize understanding over speed.

    Why bees are critical to ecosystems.
    Bees pollinate roughly one‑third of the food we consume.
    Their visits transfer pollen grains from flower to flower, enabling reproduction.
    Without bees, many crops would see dramatic yield drops.
    Beyond food, bees help maintain biodiversity in wild habitats.
    Conservation efforts include planting native flowers and reducing pesticide use.
    Educating communities about bee‑friendly practices can protect these pollinators.
    Their decline signals broader ecological distress that demands urgent action.

    The science behind a good night’s sleep.
    Sleep cycles alternate between REM and non‑REM stages.
    During non‑REM, the body repairs tissues and consolidates memory.
    REM is where vivid dreams occur, reflecting neural plasticity.
    Consistent sleep timing regulates circadian rhythms, boosting mood.
    Light exposure, especially blue light, can suppress melatonin production.
    Creating a cool, dark bedroom environment supports natural sleep onset.
    Prioritizing sleep is as essential as diet and exercise for overall health.

    The rise of urban farming.
    Cities are turning rooftops, abandoned lots, and vertical spaces into farms.
    Urban agriculture reduces food miles, lowering carbon footprints.
    Tech like hydroponics and aeroponics maximizes yield per square foot.
    Community gardens foster social cohesion and food education.
    Policies that grant access to unused land can accelerate this movement.
    Consumers benefit from fresher produce and a clearer trace of its origin.
    Urban farming exemplifies resilience in the face of climate change.

    The joy of learning a new language.
    Language learning opens doors to new cultures and perspectives.
    Even basic phrases can enhance travel experiences and local interactions.
    Consistent practice, such as daily vocabulary drills, accelerates progress.
    Listening to native media immerses you in authentic pronunciation.
    Mistakes are stepping stones; they reveal gaps in understanding.
    Language learning also sharpens cognitive flexibility and memory.
    The reward is a richer, more connected worldview.

    The magic of analog photography.
    Film cameras capture light through chemical reactions, producing unique grain textures.
    Unlike digital, each exposure is precious—there’s no instant review.
    Developing film adds a tactile, almost ritualistic element to the process.
    Color balancing on film often results in warmer, more nostalgic tones.
    The physicality of a printed image invites slower, contemplative viewing.
    Analog photography encourages intentional composition and patience.
    It reminds us that beauty can lie in imperfection and serendipity.

    The role of microbiomes in human health.
    Our gut hosts trillions of microbes that aid digestion and immunity.
    A balanced microbiome supports mental health through the gut‑brain axis.
    Dietary fibers act as prebiotics, nourishing beneficial bacteria.
    Antibiotic overuse can disrupt this delicate ecosystem.
    Probiotic foods like yogurt and kimchi introduce live cultures.
    Research suggests a link between microbiome diversity and chronic disease risk.
    Maintaining a healthy microbial community is a cornerstone of wellness.

    The power of storytelling in business.
    Stories humanize brands, turning products into relatable experiences.
    A compelling narrative can differentiate a company in a crowded market.
    Customer testimonials serve as authentic, persuasive stories.
    Storytelling aligns stakeholders around a shared vision.
    Visual elements—images, videos—enhance narrative impact.
    When employees internalize the brand story, engagement rises.
    Ultimately, stories build trust, loyalty, and lasting relationships.

    The quiet strength of introverts.
    Introverts thrive on deep focus and reflective thinking.
    Their energy is replenished by solitude rather than social interaction.
    They often become excellent listeners, absorbing details others miss.
    In collaborative settings, introverts bring thoughtful analysis.
    Balancing introverted and extroverted strengths fosters innovation.
    Understanding this dynamic can improve team cohesion.
    Embracing introversion enriches both personal and professional life.

    The evolution of electric vehicles (EVs).
    EVs have transitioned from niche to mainstream in the past decade.
    Battery technology improvements have extended range and reduced costs.
    Charging infrastructure now spans urban centers and rural highways.
    Governments offer incentives, accelerating adoption.
    EVs produce lower emissions, supporting climate goals.
    Manufacturers are investing heavily in autonomous features.
    The future of mobility is increasingly electric and data‑driven.

    The beauty of minimalist architecture.
    Minimalist design emphasizes simplicity, clean lines, and open spaces.
    Materials are chosen for their natural textures and durability.
    Large windows blur the boundary between interior and exterior.
    Neutral palettes create a calming, timeless aesthetic.
    Every element serves a purpose, eliminating visual clutter.
    Sustainability often accompanies minimalism through efficient use of resources.
    Such spaces invite mindfulness and a deeper connection to the environment.

    The importance of cyber hygiene.
    Regularly updating software patches vulnerabilities before exploitation.
    Using strong, unique passwords and a manager protects accounts.
    Two‑factor authentication adds a critical security layer.
    Be wary of phishing emails that mimic legitimate sources.
    Backups safeguard data against ransomware or hardware failure.
    Educating employees on safe practices reduces organizational risk.
    Cyber hygiene is an ongoing commitment, not a one‑time task.

    The role of music in cognitive development.
    Early exposure to music can enhance language acquisition.
    Rhythmic patterns train auditory discrimination skills.
    Instrumental practice builds fine motor coordination.
    Music therapy is used to support emotional regulation in children.
    Learning to read music develops analytical thinking.
    Cultural exposure through music broadens social understanding.
    Overall, music serves as a versatile tool for holistic growth.

    The ethics of artificial intelligence.
    AI systems can perpetuate biases present in training data.
    Transparent algorithms allow stakeholders to scrutinize decisions.
    Accountability frameworks assign responsibility for AI outcomes.
    Privacy concerns arise when AI processes personal data.
    Ethical guidelines aim to balance innovation with societal welfare.
    Ongoing dialogue between technologists, ethicists, and users is essential.
    Responsible AI ensures technology serves humanity equitably.

    The resilience of coral reefs.
    Coral reefs support over 25% of marine species in a tiny ocean area.
    They act as natural breakwaters, protecting coastlines from erosion.
    Coral bleaching occurs when water temperatures rise, stressing symbiotic algae.
    Restoration projects use micro‑fragmentation to grow new colonies.
    Local communities often rely on reefs for fisheries and tourism.
    Protecting reefs requires global cooperation and sustainable practices.
    Healthy reefs are a barometer of oceanic well‑being.

    The significance of cultural heritage sites.
    Such sites embody collective memory and identity.
    They attract tourism, generating revenue for local economies.
    Preservation efforts maintain architectural integrity and tradition.
    Community involvement ensures that conservation respects local values.
    Documentation and research safeguard intangible cultural practices.
    International charters provide guidelines for protection.
    Cultural heritage fosters continuity amid rapid modernization.

    The mechanics of a simple pendulum.
    A pendulum swings due to gravity acting on its mass.
    The period of oscillation depends on the length, not the mass.
    The restoring force is proportional to the sine of the angle.
    In a simple approximation, small angles give a constant period.
    Damping forces, like air resistance, gradually reduce amplitude.
    Pendulums historically measured time and aided in navigation.
    They remain a classic demonstration of harmonic motion.

    The value of volunteering.
    Giving time to others cultivates empathy and social responsibility.
    Volunteers often gain new skills and broaden their networks.
    Community projects address gaps that formal systems overlook.
    Volunteerism can improve mental health by fostering a sense of purpose.
    Organizations rely on volunteers for outreach and support.
    Even small acts of service ripple through society.
    Volunteering is a reciprocal investment in humanity."""
    encode_test = "hello world"
    merges = ground_llm.encode_train(text)
    encoded = ground_llm.encode(encode_test, merges)
    decoded = ground_llm.decode_string(encoded, merges)

    merges_file = ROOT / "merges_record.json"
    print(f"build_info={ground_llm.build_info()}")
    print(f"module={ground_llm.__name__}")
    print(f"text_len={len(text)}")
    print(f"text_preview={summarize_text(text)!r}")
    print(f"merge_count={len(merges)}")
    print(f"encoded={encoded}")
    print(f"last_merge_id={merges[-1][1] if merges else None}")
    print(f"decoded_ok={decoded == encode_test}")
    print(f"merges_file_exists={merges_file.exists()}")

    if merges_file.exists():
        saved = json.loads(merges_file.read_text(encoding="utf-8"))
        print(f"saved_merge_count={len(saved)}")
        if saved:
            print(f"first_saved_merge={saved[0]}")
            print(f"last_saved_merge={saved[-1]}")

    return 0 if decoded == encode_test else 1


if __name__ == "__main__":
    sys.exit(main())
