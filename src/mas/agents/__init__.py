"""Sub-agent implementations (planner, router, critic, executor, ...).

Each sub-agent is a unit of reasoning. Sub-agents never touch environment tools
directly; they emit tool *requests* that the active pattern forwards through the
adapter boundary."""
