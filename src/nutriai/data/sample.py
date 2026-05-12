"""A tiny hand-crafted sample so the pipeline runs without any download.

Five recipes, mixed cuisines, with ingredient and nutrition data so we can
exercise the parent/child splitter end-to-end.
"""

from __future__ import annotations

from src.nutriai.schemas import Nutrient, Recipe

SAMPLE_RECIPES: list[Recipe] = [
    Recipe(
        id="sample-001",
        title="Shepherd's Pie",
        ingredients=[
            "500 g ground lamb",
            "2 medium potatoes, peeled and cubed",
            "1 medium onion, finely chopped",
            "2 carrots, diced",
            "100 g frozen peas",
            "2 tbsp tomato paste",
            "250 ml beef stock",
            "30 g butter",
            "salt and pepper to taste",
        ],
        instructions=[
            "Boil the potatoes in salted water until tender, about 15 minutes.",
            "Meanwhile, brown the lamb in a large skillet over medium-high heat.",
            "Add the onion and carrots; cook until softened, about 6 minutes.",
            "Stir in the tomato paste and stock; simmer 10 minutes.",
            "Mash the potatoes with butter, salt and pepper.",
            "Spread the meat in a baking dish, top with mashed potatoes.",
            "Bake at 200 C for 20 minutes, until the top is golden.",
        ],
        nutrients=[
            Nutrient(name="Calories", amount=520.0, unit="kcal"),
            Nutrient(name="Protein", amount=28.0, unit="g"),
            Nutrient(name="Fat", amount=27.0, unit="g"),
            Nutrient(name="Carbohydrates", amount=38.0, unit="g"),
            Nutrient(name="Fiber", amount=5.0, unit="g"),
        ],
        source="sample",
    ),
    Recipe(
        id="sample-002",
        title="Quick Garlic Shrimp Pasta",
        ingredients=[
            "200 g spaghetti",
            "300 g shrimp, peeled and deveined",
            "4 garlic cloves, minced",
            "60 ml olive oil",
            "1/2 tsp red pepper flakes",
            "1 lemon, juiced",
            "fresh parsley, chopped",
            "salt to taste",
        ],
        instructions=[
            "Cook the spaghetti in salted boiling water until al dente.",
            "Heat the olive oil in a pan and saute the garlic for 30 seconds.",
            "Add the shrimp and red pepper flakes; cook until pink, about 3 minutes.",
            "Stir in the lemon juice and a splash of pasta water.",
            "Toss with the drained spaghetti and parsley; serve immediately.",
        ],
        nutrients=[
            Nutrient(name="Calories", amount=480.0, unit="kcal"),
            Nutrient(name="Protein", amount=32.0, unit="g"),
            Nutrient(name="Fat", amount=18.0, unit="g"),
            Nutrient(name="Carbohydrates", amount=50.0, unit="g"),
        ],
        source="sample",
    ),
    Recipe(
        id="sample-003",
        title="Chickpea and Spinach Curry",
        ingredients=[
            "400 g canned chickpeas, drained",
            "200 g fresh spinach",
            "1 onion, chopped",
            "3 garlic cloves, minced",
            "1 tbsp grated ginger",
            "400 ml coconut milk",
            "2 tbsp curry powder",
            "1 tbsp vegetable oil",
            "salt to taste",
        ],
        instructions=[
            "Heat the oil in a deep pan over medium heat.",
            "Saute the onion until translucent, then add garlic and ginger.",
            "Stir in the curry powder and toast for 30 seconds.",
            "Add the chickpeas and coconut milk; simmer 10 minutes.",
            "Stir in the spinach until wilted; season with salt and serve.",
        ],
        nutrients=[
            Nutrient(name="Calories", amount=410.0, unit="kcal"),
            Nutrient(name="Protein", amount=14.0, unit="g"),
            Nutrient(name="Fat", amount=23.0, unit="g"),
            Nutrient(name="Carbohydrates", amount=38.0, unit="g"),
            Nutrient(name="Fiber", amount=10.0, unit="g"),
        ],
        source="sample",
    ),
    Recipe(
        id="sample-004",
        title="Greek Yogurt Berry Parfait",
        ingredients=[
            "200 g Greek yogurt",
            "100 g mixed berries",
            "30 g granola",
            "1 tbsp honey",
            "1 tsp chia seeds",
        ],
        instructions=[
            "Layer half the yogurt in a glass.",
            "Top with half the berries and granola.",
            "Repeat the layers and finish with honey and chia seeds.",
        ],
        nutrients=[
            Nutrient(name="Calories", amount=320.0, unit="kcal"),
            Nutrient(name="Protein", amount=18.0, unit="g"),
            Nutrient(name="Fat", amount=8.0, unit="g"),
            Nutrient(name="Carbohydrates", amount=42.0, unit="g"),
            Nutrient(name="Sugar", amount=28.0, unit="g"),
        ],
        source="sample",
    ),
    Recipe(
        id="sample-005",
        title="Roasted Sweet Potato and Black Bean Bowl",
        ingredients=[
            "2 sweet potatoes, cubed",
            "400 g canned black beans, drained",
            "1 avocado, sliced",
            "150 g cooked quinoa",
            "1 lime, juiced",
            "2 tbsp olive oil",
            "1 tsp smoked paprika",
            "1/2 tsp cumin",
            "fresh cilantro, chopped",
        ],
        instructions=[
            "Toss the sweet potato cubes with olive oil, paprika and cumin.",
            "Roast at 200 C for 25 minutes, until tender and caramelised.",
            "Warm the black beans in a small pan.",
            "Build bowls with quinoa, sweet potato, beans and avocado.",
            "Finish with lime juice and cilantro.",
        ],
        nutrients=[
            Nutrient(name="Calories", amount=560.0, unit="kcal"),
            Nutrient(name="Protein", amount=18.0, unit="g"),
            Nutrient(name="Fat", amount=20.0, unit="g"),
            Nutrient(name="Carbohydrates", amount=78.0, unit="g"),
            Nutrient(name="Fiber", amount=18.0, unit="g"),
        ],
        source="sample",
    ),
]
