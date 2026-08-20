"""tau2-bench binding.

Holds the HalfDuplexAgent subclass, its factory + registry registration, and the
translation layer between tau2 message/tool types and `mas.core` types. This is
the single choke point through which every environment tool call must pass."""
