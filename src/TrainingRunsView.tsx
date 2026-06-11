import React, { useState, useCallback, useEffect, useRef } from "react";
import { useTriggerPanelEvent } from "@fiftyone/operators";
import {
  Box,
  Button,
  Card,
  CardActionArea,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  IconButton,
  Link,
  Menu,
  MenuItem,
  Select,
  Stack,
  SvgIcon,
  Tab,
  Table,
  TableBody,
  TableRow,
  TableCell,
  Tabs,
  TextField,
  Tooltip,
  Typography,
  styled,
} from "@mui/material";

const SPLITS = ["train", "val", "test"] as const;
type Split = (typeof SPLITS)[number];

interface SplitInfo {
  present: boolean;
  label?: string | null;
  num_samples?: number;
}

interface SplitDetail {
  num_samples: number;
  stages: string[];
  label_field?: string | null;
  distribution?: { label: string; count: number }[] | null;
}

interface Row {
  train_key: string;
  checkpoint_uri?: string | null;
  project_url?: string | null;
  eval_key?: string | null;
  label_field?: string | null;
  created_at?: string | null;
  status?: string;
  exec_status?: string;
  train_config?: Record<string, any> | null;
  note?: string;
  splits?: Partial<Record<Split, SplitInfo>>;
}

interface Props {
  data?: {
    rows?: Row[];
    statuses?: Record<string, string>;
    eval_summary?: EvalSummary & { train_key?: string };
    split_details?: {
      train_key?: string;
      splits?: Record<string, SplitDetail>;
    };
  };
  schema?: { view?: Record<string, string> };
}

const STATUS_COLOR: Record<string, string> = {
  new: "#999999",
  candidate: "#FFB682",
  promoted: "#8BC18D",
  archived: "#82AAFF",
};
const DEFAULT_STATUSES: Record<string, string> = {
  new: "New",
  candidate: "Candidate",
  promoted: "Promoted",
  archived: "Archived",
};

// Execution lifecycle (config.status), distinct from the review pill above.
const EXEC_STATUS_COLOR: Record<string, string> = {
  declared: "#999999",
  scheduled: "#82AAFF",
  queued: "#82AAFF",
  running: "#FFB682",
  in_progress: "#FFB682",
  completed: "#8BC18D",
  failed: "#FF6B6B",
};
const EXEC_STATUS_LABEL: Record<string, string> = {
  declared: "Declared",
  scheduled: "Scheduled",
  queued: "Queued",
  running: "Running",
  in_progress: "Running",
  completed: "Completed",
  failed: "Failed",
};
const ORANGE = "#FF6D04";

const InfoTable = styled(Table)(({ theme }) => ({
  ".MuiTableCell-root": {
    border: `1px solid ${theme.palette.divider}`,
    fontSize: 14,
    verticalAlign: "top",
  },
}));

const tabsSx: any = {
  minHeight: 36,
  flexShrink: 0,
  // size to the tabs' content (don't stretch the bar to full panel width)
  "& .MuiTabs-flexContainer": { height: 36, display: "inline-flex" },
  "& .MuiTab-root": {
    minHeight: 36,
    height: 36,
    padding: "7px 16px",
    minWidth: 140,
    textTransform: "none",
    border: "1px solid",
    borderColor: (theme: any) => theme.palette.divider,
    borderRight: "none",
    "&:first-of-type": { borderTopLeftRadius: 4, borderBottomLeftRadius: 4 },
    "&:last-of-type": {
      borderTopRightRadius: 4,
      borderBottomRightRadius: 4,
      borderRight: "1px solid",
      borderColor: (theme: any) => theme.palette.divider,
    },
    "&.Mui-selected": {
      color: (theme: any) => theme.palette.text.primary,
      backgroundColor: (theme: any) =>
        theme.palette.background?.button || theme.palette.action.selected,
    },
  },
};

function RunIcon(props: any) {
  // "directions_run" — a person running
  return (
    <SvgIcon {...props} viewBox="0 0 24 24">
      <path d="M13.49 5.48c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm-3.6 13.9l1-4.4 2.1 2v6h2v-7.5l-2.1-2 .6-3c1.3 1.5 3.3 2.5 5.5 2.5v-2c-1.9 0-3.5-1-4.3-2.4l-1-1.6c-.4-.6-1-1-1.7-1-.3 0-.5.1-.8.1l-5.2 2.2v4.7h2v-3.4l1.8-.7-1.6 8.1-4.9-1-.4 2 7 1.4z" />
    </SvgIcon>
  );
}
function BackIcon(props: any) {
  return (
    <SvgIcon {...props} viewBox="0 0 24 24">
      <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z" />
    </SvgIcon>
  );
}
function AddIcon(props: any) {
  return (
    <SvgIcon {...props} viewBox="0 0 24 24">
      <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
    </SvgIcon>
  );
}
function MoreVertIcon(props: any) {
  return (
    <SvgIcon {...props} viewBox="0 0 24 24">
      <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z" />
    </SvgIcon>
  );
}
function Dot({ color }: { color: string }) {
  return (
    <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: color, flexShrink: 0 }} />
  );
}

function StatusPill({ status, label }: { status: string; label: string }) {
  const color = STATUS_COLOR[status] || STATUS_COLOR.new;
  return (
    <Chip
      size="small"
      variant="filled"
      label={
        <Stack direction="row" spacing={0.5} sx={{ alignItems: "center" }}>
          <Dot color={color} />
          <Typography sx={{ color, fontSize: 13 }}>{label}</Typography>
        </Stack>
      }
      sx={{
        border: "none",
        backgroundColor: (theme: any) => `${theme.palette.action.selected}!important`,
      }}
    />
  );
}

function ExecStatusPill({ status }: { status?: string }) {
  const s = status || "declared";
  const color = EXEC_STATUS_COLOR[s] || EXEC_STATUS_COLOR.declared;
  const label = EXEC_STATUS_LABEL[s] || s;
  return (
    <Chip
      size="small"
      variant="outlined"
      label={
        <Stack direction="row" spacing={0.5} sx={{ alignItems: "center" }}>
          <Dot color={color} />
          <Typography sx={{ color, fontSize: 13 }}>{label}</Typography>
        </Stack>
      }
      sx={{ borderColor: color }}
    />
  );
}

function formatCreated(iso?: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function Muted({ children }: { children: React.ReactNode }) {
  return (
    <Typography component="span" sx={{ opacity: 0.5 }}>
      {children}
    </Typography>
  );
}

function LinkValue({ url }: { url?: string | null }) {
  if (!url) return <Muted>—</Muted>;
  return (
    <Tooltip title={url} arrow>
      <Link
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        underline="hover"
        sx={{
          display: "inline-block",
          maxWidth: 420,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          verticalAlign: "bottom",
        }}
      >
        {url}
      </Link>
    </Tooltip>
  );
}

function PropRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <TableRow>
      <TableCell sx={{ width: 180, color: "text.secondary", whiteSpace: "nowrap" }}>
        {label}
      </TableCell>
      <TableCell>{children}</TableCell>
    </TableRow>
  );
}

function StagesList({ stages }: { stages?: string[] }) {
  if (!stages || stages.length === 0)
    return <Muted>Entire dataset (no view stages).</Muted>;
  return (
    <Stack component="ol" spacing={0.25} sx={{ pl: 2.5, m: 0 }}>
      {stages.map((s, i) => (
        <Typography
          key={i}
          component="li"
          variant="caption"
          sx={{ fontFamily: "monospace", wordBreak: "break-all" }}
        >
          {s}
        </Typography>
      ))}
    </Stack>
  );
}

function DistributionBars({
  dist,
  labelField,
  loading,
}: {
  dist?: { label: string; count: number }[] | null;
  labelField?: string | null;
  loading?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  if (!labelField)
    return <Muted>No label field recorded for this run.</Muted>;
  if (loading) return <Muted>Loading…</Muted>;
  if (!dist || dist.length === 0) return <Muted>No labels found.</Muted>;
  const shown = expanded ? dist : dist.slice(0, 15);
  const max = Math.max(...dist.map((d) => d.count), 1);
  return (
    <Stack spacing={0.5}>
      {shown.map((d) => (
        <Stack key={d.label} direction="row" spacing={1} sx={{ alignItems: "center" }}>
          <Typography
            variant="caption"
            title={d.label}
            sx={{
              width: 130,
              flexShrink: 0,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {d.label}
          </Typography>
          <Box
            sx={{
              flex: 1,
              height: 14,
              bgcolor: "action.hover",
              borderRadius: 0.5,
              overflow: "hidden",
            }}
          >
            <Box sx={{ width: `${(d.count / max) * 100}%`, height: "100%", bgcolor: ORANGE }} />
          </Box>
          <Typography variant="caption" sx={{ width: 52, textAlign: "right", flexShrink: 0 }}>
            {d.count}
          </Typography>
        </Stack>
      ))}
      {dist.length > 15 && (
        <Button
          size="small"
          onClick={() => setExpanded((e) => !e)}
          sx={{ alignSelf: "flex-start", textTransform: "none", px: 0.5 }}
        >
          {expanded ? "Show fewer classes" : `+${dist.length - 15} more classes`}
        </Button>
      )}
    </Stack>
  );
}

function ConfigValue({ value }: { value: any }) {
  // A long/multiline string is treated as a script; objects are pretty-printed;
  // scalars render inline.
  const isScript =
    typeof value === "string" && (value.includes("\n") || value.length > 200);
  if (isScript || (typeof value === "object" && value !== null)) {
    const text =
      typeof value === "string" ? value : JSON.stringify(value, null, 2);
    return (
      <Box
        component="pre"
        sx={{
          m: 0,
          fontFamily: "monospace",
          fontSize: 12,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {text}
      </Box>
    );
  }
  if (value === null || value === undefined || value === "")
    return <Muted>—</Muted>;
  return <span>{String(value)}</span>;
}

function TrainingConfigSection({
  config,
}: {
  config?: Record<string, any> | null;
}) {
  const [open, setOpen] = useState(false);
  const entries = config ? Object.entries(config) : [];
  // A custom-script run carries its source under `script`; everything else is
  // shown as a parameter table (the builtin "Train model" case).
  const script =
    config && typeof config.script === "string" ? config.script : null;
  const params = entries.filter(([k]) => k !== "script");

  return (
    <Box>
      <Stack
        direction="row"
        spacing={0.5}
        onClick={() => setOpen((o) => !o)}
        sx={{ alignItems: "center", cursor: "pointer", userSelect: "none" }}
      >
        <Typography sx={{ width: 16, color: "text.secondary" }}>
          {open ? "▾" : "▸"}
        </Typography>
        <Typography variant="body1" color="secondary">
          Training configuration
        </Typography>
      </Stack>
      {open && (
        <Box sx={{ mt: 1 }}>
          {entries.length === 0 ? (
            <Muted>No training configuration recorded for this run.</Muted>
          ) : (
            <Stack spacing={1.5}>
              {script !== null && (
                <Box>
                  <Typography variant="overline" color="secondary">
                    Training script
                  </Typography>
                  <Box
                    sx={{
                      mt: 0.5,
                      p: 1,
                      borderRadius: 1,
                      bgcolor: "action.hover",
                      maxHeight: 360,
                      overflow: "auto",
                    }}
                  >
                    <ConfigValue value={script} />
                  </Box>
                </Box>
              )}
              {params.length > 0 && (
                <InfoTable size="small" sx={{ width: "auto", maxWidth: 820 }}>
                  <TableBody>
                    {params.map(([k, v]) => (
                      <TableRow key={k}>
                        <TableCell
                          sx={{
                            width: 180,
                            color: "text.secondary",
                            whiteSpace: "nowrap",
                          }}
                        >
                          {k}
                        </TableCell>
                        <TableCell>
                          <ConfigValue value={v} />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </InfoTable>
              )}
            </Stack>
          )}
        </Box>
      )}
    </Box>
  );
}

interface EvalSummary {
  eval_key?: string;
  type?: string;
  method?: string;
  gt_field?: string;
  pred_field?: string;
  metrics?: Record<string, number> | null;
  error?: string;
}

const METRIC_LABELS: Record<string, string> = {
  mAP: "mAP",
  precision: "Precision",
  recall: "Recall",
  fscore: "F1",
  accuracy: "Accuracy",
  support: "Support",
  tp: "TP",
  fp: "FP",
  fn: "FN",
};
const METRIC_ORDER = ["mAP", "precision", "recall", "fscore", "accuracy", "support", "tp", "fp", "fn"];

function EvalSummaryView({ summary }: { summary?: EvalSummary | null }) {
  if (summary === undefined || summary === null) return <Muted>Loading…</Muted>;
  const m = summary.metrics;
  if (summary.error || !m || Object.keys(m).length === 0)
    return <Muted>No metrics available for this eval.</Muted>;

  const keys = [
    ...METRIC_ORDER.filter((k) => k in m),
    ...Object.keys(m).filter((k) => !METRIC_ORDER.includes(k)),
  ];
  const fmt = (v: number) =>
    typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : String(v);

  return (
    <Stack spacing={1}>
      <InfoTable size="small" sx={{ width: "auto", maxWidth: 820 }}>
        <TableBody>
          {keys.map((k) => (
            <TableRow key={k}>
              <TableCell sx={{ width: 180, color: "text.secondary", whiteSpace: "nowrap" }}>
                {METRIC_LABELS[k] || k}
              </TableCell>
              <TableCell>{fmt(m[k])}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </InfoTable>
      {(summary.type || summary.method) && (
        <Typography variant="caption" color="secondary">
          {summary.type}
          {summary.method ? ` · ${summary.method}` : ""}
          {summary.pred_field ? ` · pred: ${summary.pred_field}` : ""}
          {summary.gt_field ? ` · gt: ${summary.gt_field}` : ""}
        </Typography>
      )}
    </Stack>
  );
}

function NoteEditor({
  trainKey,
  initial,
  onSave,
}: {
  trainKey: string;
  initial: string;
  onSave: (note: string) => void;
}) {
  const [text, setText] = useState(initial);
  useEffect(() => setText(initial), [trainKey, initial]);
  const dirty = text !== initial;
  return (
    <Stack spacing={1}>
      <TextField
        multiline
        minRows={2}
        size="small"
        value={text}
        placeholder="No notes added yet. Add a note about this run…"
        onChange={(e) => setText(e.target.value)}
      />
      <Box>
        <Button size="small" variant="outlined" disabled={!dirty} onClick={() => onSave(text)}>
          Save note
        </Button>
      </Box>
    </Stack>
  );
}

export default function TrainingRunsView({ data, schema }: Props) {
  const triggerEvent = useTriggerPanelEvent();
  const view = schema?.view ?? {};
  const rows: Row[] = data?.rows ?? [];
  const statuses = data?.statuses ?? DEFAULT_STATUSES;
  const statusLabel = (s: string) => statuses[s] || s;

  const [selected, setSelected] = useState<string | null>(null);
  const [tab, setTab] = useState("overview");
  const evalSummary = data?.eval_summary;
  const requestedEvalFor = useRef<string | null>(null);
  const requestedDetailsFor = useRef<string | null>(null);
  const [menuAnchor, setMenuAnchor] = useState<HTMLElement | null>(null);
  const [menuRun, setMenuRun] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  // `useTriggerPanelEvent` (the hook the builtin Model Evaluation panel uses)
  // dispatches the method AND syncs the panel's `set_data` writes back to this
  // component, so methods can push results via `ctx.panel.set_data` and we read
  // them off the `data` prop. (Plain `usePanelEvent`/`handleEvent` does not sync.)
  const call = useCallback(
    (name: string, uri?: string, params: Record<string, unknown> = {}) => {
      if (!uri) {
        console.warn("[tr][js] NO METHOD URI for", name, "— view keys:", Object.keys(view));
        return;
      }
      triggerEvent(uri, params);
    },
    [triggerEvent, view]
  );

  const refresh = () => call("refresh", view.refresh);
  const openLog = () => call("open_log", view.open_log);
  const openEdit = (trainKey: string) =>
    call("open_edit", view.open_edit, { train_key: trainKey });
  const deleteRun = (trainKey: string) =>
    call("delete_run", view.delete_run, { train_key: trainKey });
  const openEval = (evalKey?: string | null) =>
    evalKey && call("open_eval", view.open_eval, { eval_key: evalKey });
  const openView = (trainKey: string, split: Split) =>
    call("open_view", view.open_view, { train_key: trainKey, split });
  const setStatus = (trainKey: string, status: string) =>
    call("set_status", view.set_status, { train_key: trainKey, status });
  const setNote = (trainKey: string, note: string) =>
    call("set_note", view.set_note, { train_key: trainKey, note });

  const current = selected ? rows.find((r) => r.train_key === selected) : undefined;

  // Reset to the overview tab when switching runs.
  useEffect(() => {
    setTab("overview");
  }, [selected]);

  const openMenu = (e: React.MouseEvent<HTMLElement>, trainKey: string) => {
    setMenuAnchor(e.currentTarget);
    setMenuRun(trainKey);
  };
  const closeMenu = () => setMenuAnchor(null);

  const overlays = (
    <>
      <Menu anchorEl={menuAnchor} open={Boolean(menuAnchor)} onClose={closeMenu}>
        <MenuItem
          onClick={() => {
            setDeleteTarget(menuRun);
            closeMenu();
          }}
          sx={{ color: "error.main" }}
        >
          Delete training run
        </MenuItem>
      </Menu>
      <Dialog open={Boolean(deleteTarget)} onClose={() => setDeleteTarget(null)}>
        <DialogTitle>Delete training run?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the training run <b>{deleteTarget}</b>? This
            action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteTarget(null)}>Cancel</Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => {
              const t = deleteTarget;
              if (t) {
                deleteRun(t);
                if (selected === t) setSelected(null);
              }
              setDeleteTarget(null);
            }}
          >
            Delete run
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );

  // Lazily request per-split details when the Data splits tab is opened.
  // Python pushes the result back via `set_data("split_details", ...)`.
  useEffect(() => {
    if (!selected || tab !== "data") return;
    if (requestedDetailsFor.current === selected) return;
    requestedDetailsFor.current = selected;
    call("split_details", view.split_details, { train_key: selected });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, tab]);

  // The pushed split details are for whichever run we last requested; only use
  // them once they match the run currently open.
  const details =
    selected && data?.split_details?.train_key === selected
      ? data.split_details.splits ?? {}
      : null;

  // Request the eval metrics summary for the selected run (fire-and-forget;
  // Python pushes the result back via panel state, so there's no hanging
  // callback even when several runs share an eval_key).
  useEffect(() => {
    const evalKey = current?.eval_key;
    if (!selected || !evalKey) return;
    if (requestedEvalFor.current === selected) return;
    requestedEvalFor.current = selected;
    call("eval_summary", view.eval_summary, { train_key: selected, eval_key: evalKey });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, current?.eval_key]);

  // The pushed summary is for whichever run we last requested; only show it once
  // it matches the run currently open.
  const currentEvalSummary =
    selected && evalSummary?.train_key === selected ? evalSummary : undefined;

  // ---- detail view -------------------------------------------------------
  if (selected && current) {
    const r = current;
    const status = r.status || "new";
    const presentSplits = SPLITS.filter((s) => r.splits?.[s]?.present);
    const loadingDetails = details === null;
    return (
      <Stack spacing={2} sx={{ p: 2, height: "100%", overflow: "auto" }}>
        <Stack direction="row" sx={{ justifyContent: "space-between" }}>
          <Stack direction="row" spacing={1} sx={{ alignItems: "center", flex: 1, flexWrap: "wrap" }}>
            <IconButton size="small" onClick={() => setSelected(null)}>
              <BackIcon fontSize="small" />
            </IconButton>
            <RunIcon sx={{ color: ORANGE }} fontSize="small" />
            <Typography sx={{ fontSize: 18, fontWeight: 600 }}>{r.train_key}</Typography>
            <ExecStatusPill status={r.exec_status} />
            <Select
              value={status}
              onChange={(e) => setStatus(r.train_key, String(e.target.value))}
              sx={{
                height: 28,
                borderRadius: "16px",
                backgroundColor: (theme: any) => theme.palette.action.selected,
                "& fieldset": { border: "none" },
              }}
            >
              {Object.keys(statuses).map((k) => (
                <MenuItem key={k} value={k}>
                  <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                    <Dot color={STATUS_COLOR[k] || STATUS_COLOR.new} />
                    <Typography sx={{ color: STATUS_COLOR[k] || STATUS_COLOR.new }}>
                      {statusLabel(k)}
                    </Typography>
                  </Stack>
                </MenuItem>
              ))}
            </Select>
          </Stack>
          <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
            <Button size="small" variant="outlined" onClick={() => openEdit(r.train_key)}>
              Edit
            </Button>
            <IconButton size="small" onClick={(e) => openMenu(e, r.train_key)}>
              <MoreVertIcon fontSize="small" />
            </IconButton>
          </Stack>
        </Stack>
        {overlays}

        <Tabs
          value={tab}
          onChange={(_, v) => setTab(v)}
          sx={tabsSx}
          TabIndicatorProps={{ style: { display: "none" } }}
        >
          <Tab value="overview" label="Overview" />
          <Tab value="data" label="Data splits" />
        </Tabs>

        {tab === "overview" && (
          <Stack spacing={2}>
            <InfoTable size="small" sx={{ width: "auto", maxWidth: 820 }}>
              <TableBody>
                <PropRow label="Model checkpoint">
                  <LinkValue url={r.checkpoint_uri} />
                </PropRow>
                <PropRow label="Experiment / tracker">
                  <LinkValue url={r.project_url} />
                </PropRow>
                <PropRow label="Label field">
                  {r.label_field ? <code>{r.label_field}</code> : <Muted>— not recorded</Muted>}
                </PropRow>
                <PropRow label="Eval run">
                  {r.eval_key ? <code>{r.eval_key}</code> : <Muted>— not linked</Muted>}
                </PropRow>
                <PropRow label="Created">{formatCreated(r.created_at)}</PropRow>
              </TableBody>
            </InfoTable>

            <TrainingConfigSection config={r.train_config} />

            {r.eval_key && (
              <Box>
                <Stack
                  direction="row"
                  spacing={1}
                  sx={{ mb: 1, alignItems: "center" }}
                >
                  <Typography variant="body1" color="secondary">
                    Evaluation summary
                  </Typography>
                  <Button size="small" onClick={() => openEval(r.eval_key)}>
                    Open eval ↗
                  </Button>
                </Stack>
                <EvalSummaryView summary={currentEvalSummary} />
              </Box>
            )}

            <Box>
              <Typography variant="body1" color="secondary" sx={{ mb: 1 }}>
                Notes
              </Typography>
              <NoteEditor
                trainKey={r.train_key}
                initial={r.note || ""}
                onSave={(note) => setNote(r.train_key, note)}
              />
            </Box>
          </Stack>
        )}

        {tab === "data" && (
          <Stack spacing={2}>
            {presentSplits.length === 0 && <Muted>No data splits recorded for this run.</Muted>}
            {presentSplits.map((s) => {
              const summary = r.splits?.[s];
              const detail = details?.[s];
              const count = detail?.num_samples ?? summary?.num_samples;
              return (
                <Card key={s} variant="outlined" sx={{ p: 2 }}>
                  <Stack
                    direction="row"
                    sx={{ justifyContent: "space-between", alignItems: "center", mb: 1 }}
                  >
                    <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                      <Typography sx={{ fontSize: 15, fontWeight: 600 }}>
                        {s[0].toUpperCase() + s.slice(1)}
                      </Typography>
                      {summary?.label && (
                        <Chip size="small" variant="outlined" label={summary.label} />
                      )}
                      {typeof count === "number" && (
                        <Typography variant="caption" color="secondary">
                          {count} samples
                        </Typography>
                      )}
                    </Stack>
                    <Button size="small" onClick={() => openView(r.train_key, s)}>
                      Open in grid
                    </Button>
                  </Stack>

                  <Typography variant="overline" color="secondary">
                    View stages
                  </Typography>
                  <Box sx={{ mb: 1.5 }}>
                    {loadingDetails ? <Muted>Loading…</Muted> : <StagesList stages={detail?.stages} />}
                  </Box>

                  <Typography variant="overline" color="secondary">
                    Label distribution{r.label_field ? ` · ${r.label_field}` : ""}
                  </Typography>
                  <Box sx={{ mt: 0.5 }}>
                    <DistributionBars
                      dist={detail?.distribution}
                      labelField={r.label_field}
                      loading={loadingDetails}
                    />
                  </Box>
                </Card>
              );
            })}
          </Stack>
        )}
      </Stack>
    );
  }

  // ---- list view ---------------------------------------------------------
  return (
    <Stack spacing={2} sx={{ p: 2, height: "100%", overflow: "auto" }}>
      {overlays}
      <Stack direction="row" sx={{ justifyContent: "space-between", alignItems: "center" }}>
        <Typography variant="body1" color="secondary">
          {rows.length} Training Run{rows.length === 1 ? "" : "s"}
        </Typography>
        <Stack direction="row" spacing={1}>
          <Button
            variant="contained"
            size="small"
            onClick={openLog}
            startIcon={<AddIcon fontSize="small" />}
            sx={{
              backgroundColor: ORANGE,
              textTransform: "none",
              "&:hover": { backgroundColor: ORANGE },
            }}
          >
            Log run
          </Button>
          <Button size="small" variant="outlined" onClick={refresh}>
            Refresh
          </Button>
        </Stack>
      </Stack>

      {rows.length === 0 ? (
        <Typography color="secondary">
          No training runs recorded yet. Click <b>Log run</b> to add one, or call{" "}
          <code>add_training_run(...)</code> from a notebook.
        </Typography>
      ) : (
        rows.map((r) => {
          const present = SPLITS.filter((s) => r.splits?.[s]?.present);
          const status = r.status || "new";
          return (
            <CardActionArea key={r.train_key}>
              <Card sx={{ p: 2, cursor: "pointer" }} onClick={() => setSelected(r.train_key)}>
                <Stack direction="row" sx={{ justifyContent: "space-between", alignItems: "center" }}>
                  <Stack direction="row" spacing={0.75} sx={{ alignItems: "center" }}>
                    <RunIcon sx={{ color: ORANGE }} fontSize="small" />
                    <Typography sx={{ fontSize: 16, fontWeight: 600 }}>{r.train_key}</Typography>
                  </Stack>
                  <Stack direction="row" spacing={0.5} sx={{ alignItems: "center" }}>
                    {r.eval_key && <Chip size="small" variant="outlined" label="eval" />}
                    <ExecStatusPill status={r.exec_status} />
                    <StatusPill status={status} label={statusLabel(status)} />
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        openMenu(e, r.train_key);
                      }}
                    >
                      <MoreVertIcon fontSize="small" />
                    </IconButton>
                  </Stack>
                </Stack>
                <Stack direction="row" spacing={0.5} sx={{ mt: 1, alignItems: "center" }}>
                  {present.map((s) => (
                    <Chip key={s} size="small" variant="outlined" label={s} />
                  ))}
                  {typeof r.splits?.train?.num_samples === "number" && (
                    <Typography variant="caption" color="secondary" sx={{ ml: 0.5 }}>
                      {r.splits.train.num_samples} train samples
                    </Typography>
                  )}
                  <Box sx={{ flex: 1 }} />
                  <Typography variant="caption" color="secondary">
                    {formatCreated(r.created_at)}
                  </Typography>
                </Stack>
                {r.note && (
                  <Typography variant="body2" color="secondary" noWrap sx={{ mt: 1 }}>
                    {r.note}
                  </Typography>
                )}
              </Card>
            </CardActionArea>
          );
        })
      )}
    </Stack>
  );
}
