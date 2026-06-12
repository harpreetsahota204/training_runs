import { PluginComponentType, registerComponent } from "@fiftyone/plugins";
import TrainingRunsView from "./TrainingRunsView";

// Rendered by the Python TrainingRunsPanel via composite_view; the name must
// match the `component=` kwarg in panel.py's render().
registerComponent({
  name: "TrainingRunsView",
  component: TrainingRunsView,
  type: PluginComponentType.Component,
});
