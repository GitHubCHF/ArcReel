
import { useState, useEffect, useCallback, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Loader2 } from "lucide-react";
import { useWarnUnsaved } from "@/hooks/useWarnUnsaved";
import { API } from "@/api";
import type { SystemConfigSettings, SystemConfigOptions, SystemConfigPatch } from "@/types/system";
import type { CustomProviderInfo } from "@/types/custom-provider";
import { ProviderModelSelect } from "@/components/ui/ProviderModelSelect";
import { ImageModelDualSelect } from "@/components/shared/ImageModelDualSelect";
import { PROVIDER_NAMES } from "@/components/ui/ProviderIcon";
import { useAppStore } from "@/stores/app-store";
import { useConfigStatusStore } from "@/stores/config-status-store";
import { errMsg } from "@/utils/async";
import { getCustomProviderModels } from "@/utils/provider-models";
import { ACCENT_BTN_CLS, ACCENT_BUTTON_STYLE, CARD_STYLE } from "@/components/ui/darkroom-tokens";

interface CardProps {
  kicker: string;
  title?: string;
  description?: string;
  children: React.ReactNode;
}

function SectionCard({ kicker, title, description, children }: CardProps) {
  return (
    <div
      className="rounded-[10px] border border-hairline p-5"
      style={CARD_STYLE}
    >
      <div className="mb-4">
        <div className="font-mono text-[10px] font-bold uppercase tracking-[0.16em] text-accent-2">
          {kicker}
        </div>
        {title && (
          <h4 className="mt-1.5 text-[14px] font-medium text-text">{title}</h4>
        )}
        {description && (
          <p className="mt-1 text-[12px] leading-[1.55] text-text-3">{description}</p>
        )}
      </div>
      {children}
    </div>
  );
}

export function MediaModelSection() {
  const { t } = useTranslation("dashboard");

  const TEXT_MODEL_FIELDS = useMemo(
    () =>
      [
        ["text_backend_script", t("script_generation")],
        ["text_backend_overview", t("overview_generation")],
        ["text_backend_style", t("style_analysis")],
      ] as const,
    [t],
  );

  const [settings, setSettings] = useState<SystemConfigSettings | null>(null);
  const [options, setOptions] = useState<SystemConfigOptions | null>(null);
  const [customProviders, setCustomProviders] = useState<CustomProviderInfo[]>([]);
  const [draft, setDraft] = useState<SystemConfigPatch>({});
  const [saving, setSaving] = useState(false);
  const [clearingCache, setClearingCache] = useState(false);

  const isDirty = Object.keys(draft).length > 0;
  useWarnUnsaved(isDirty);

  const allProviderNames = useMemo(
    () => ({ ...PROVIDER_NAMES, ...(options?.provider_names ?? {}) }),
    [options],
  );

  const fetchConfig = useCallback(async () => {
    const [res, custom] = await Promise.all([
      API.getSystemConfig(),
      getCustomProviderModels().catch(() => [] as CustomProviderInfo[]),
    ]);
    setSettings(res.settings);
    setOptions(res.options);
    setCustomProviders(custom);
    setDraft({});
  }, []);

  useEffect(() => {
    // mount/依赖变更时异步拉取配置，回调内 setSettings 等（异步 fetch 后回写）
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void fetchConfig();
  }, [fetchConfig]);

  const handleSave = useCallback(async () => {
    if (Object.keys(draft).length === 0) return;
    setSaving(true);
    try {
      await API.updateSystemConfig(draft);
      await fetchConfig();
      void useConfigStatusStore.getState().refresh();
      useAppStore.getState().pushToast(t("media_config_saved"), "success");
    } catch (err) {
      useAppStore.getState().pushToast(t("save_failed", { message: errMsg(err) }), "error");
    } finally {
      setSaving(false);
    }
  }, [draft, fetchConfig, t]);

  const handleClearCache = useCallback(async () => {
    setClearingCache(true);
    try {
      const { deleted } = await API.clearProviderAssetCache();
      useAppStore.getState().pushToast(t("asset_cache_cleared", { count: deleted }), "success");
    } catch (err) {
      useAppStore.getState().pushToast(t("save_failed", { message: errMsg(err) }), "error");
    } finally {
      setClearingCache(false);
    }
  }, [t]);

  if (!settings || !options) {
    return (
      <div className="flex items-center gap-2 px-1 py-12 text-text-3">
        <Loader2 className="h-3.5 w-3.5 motion-safe:animate-spin text-accent-2" aria-hidden />
        <span className="font-mono text-[11px] uppercase tracking-[0.14em]">
          {t("common:loading")}
        </span>
      </div>
    );
  }

  const videoBackends: string[] = options.video_backends ?? [];
  const imageBackends: string[] = options.image_backends ?? [];
  const textBackends: string[] = options.text_backends ?? [];
  const audioBackends: string[] = options.audio_backends ?? [];

  const currentVideo = draft.default_video_backend ?? settings.default_video_backend ?? "";
  const currentImageT2I =
    draft.default_image_backend_t2i ??
    settings.default_image_backend_t2i ??
    settings.default_image_backend ??
    "";
  const currentImageI2I =
    draft.default_image_backend_i2i ??
    settings.default_image_backend_i2i ??
    settings.default_image_backend ??
    "";
  const currentAudio = draft.video_generate_audio ?? settings.video_generate_audio ?? false;
  const currentAudioBackend = draft.default_audio_backend ?? settings.default_audio_backend ?? "";
  const currentNarrationVoice = draft.narration_voice ?? settings.narration_voice ?? "";
  const currentNarrationSpeed =
    "narration_speed" in draft ? draft.narration_speed : settings.narration_speed;
  const currentAssetCacheMode =
    draft.provider_asset_cache_mode ?? settings.provider_asset_cache_mode ?? "cached";

  const ossInputCls =
    "w-full rounded-[8px] border border-hairline bg-bg-grad-a/55 px-3 py-2 text-[12.5px] text-text placeholder:text-text-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent";
  const ossField = (
    key: "oss_endpoint" | "oss_bucket" | "oss_access_key_id" | "oss_upload_prefix",
    label: string,
    placeholder: string,
  ) => (
    <div>
      <div className="mb-1.5 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-text-4">{label}</div>
      <input
        type="text"
        value={draft[key] ?? settings[key] ?? ""}
        placeholder={placeholder}
        onChange={(e) => setDraft((prev) => ({ ...prev, [key]: e.target.value }))}
        className={ossInputCls}
        aria-label={label}
      />
    </div>
  );

  const emptyHint = (msg: string) => (
    <div className="rounded-[8px] border border-hairline-soft bg-bg-grad-a/45 px-3 py-2.5 text-[12px] text-text-3">
      {msg}
    </div>
  );

  return (
    <div className="space-y-7">
      {/* Heading */}
      <div>
        <div className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-accent-2">
          Default Routing
        </div>
        <h3
          className="font-editorial mt-1"
          style={{
            fontWeight: 400,
            fontSize: 22,
            lineHeight: 1.1,
            letterSpacing: "-0.012em",
            color: "var(--color-text)",
          }}
        >
          {t("model_selection")}
        </h3>
        <p className="mt-1.5 text-[12.5px] leading-[1.6] text-text-3">
          {t("model_selection_desc")}
        </p>
      </div>

      {/* Video */}
      <SectionCard kicker="Video Channel" title={t("default_video_model")}>
        {videoBackends.length > 0 ? (
          <ProviderModelSelect
            value={currentVideo}
            options={videoBackends}
            providerNames={allProviderNames}
            onChange={(v) => setDraft((prev) => ({ ...prev, default_video_backend: v }))}
            allowDefault
            defaultLabel={t("auto_select")}
            defaultHint={t("auto")}
          />
        ) : (
          emptyHint(t("no_video_providers_hint"))
        )}

        <div className="mt-4 flex items-start gap-2.5 text-[12.5px] text-text-2">
          <input
            id="media-generate-audio"
            type="checkbox"
            checked={currentAudio}
            onChange={(e) =>
              setDraft((prev) => ({ ...prev, video_generate_audio: e.target.checked }))
            }
            className="mt-0.5 h-3.5 w-3.5 cursor-pointer rounded border-hairline bg-bg-grad-a accent-[var(--color-accent)]"
          />
          <label htmlFor="media-generate-audio" className="flex cursor-pointer flex-col">
            <span>{t("generate_audio")}</span>
            <span className="text-[11px] text-text-4">{t("audio_support_hint")}</span>
          </label>
        </div>
      </SectionCard>

      {/* Image */}
      <SectionCard kicker="Image Channel" title={t("default_image_model")}>
        {imageBackends.length > 0 ? (
          <ImageModelDualSelect
            valueT2I={currentImageT2I}
            valueI2I={currentImageI2I}
            options={imageBackends}
            providerNames={allProviderNames}
            customProviders={customProviders}
            onChange={({ t2i, i2i }) =>
              setDraft((prev) => ({
                ...prev,
                default_image_backend_t2i: t2i,
                default_image_backend_i2i: i2i,
              }))
            }
            labelPrimary={t("default_image_model")}
            labelT2I={t("image_model_t2i")}
            labelI2I={t("image_model_i2i")}
            defaultLabel={t("auto_select")}
            defaultHint={t("auto")}
            showCapabilityHint={false}
          />
        ) : (
          emptyHint(t("no_image_providers_hint"))
        )}
      </SectionCard>

      {/* Text */}
      <SectionCard kicker="Text Channel" title={t("text_models")} description={t("text_models_desc")}>
        {textBackends.length > 0 ? (
          <div className="space-y-3.5">
            {TEXT_MODEL_FIELDS.map(([key, label]) => (
              <div key={key}>
                <div className="mb-1.5 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-text-4">
                  {label}
                </div>
                <ProviderModelSelect
                  value={draft[key] ?? settings[key] ?? ""}
                  options={textBackends}
                  providerNames={allProviderNames}
                  onChange={(v) => setDraft((prev) => ({ ...prev, [key]: v }))}
                  allowDefault
                  defaultHint={t("auto")}
                  aria-label={label}
                />
              </div>
            ))}
          </div>
        ) : (
          emptyHint(t("no_text_providers_hint"))
        )}
      </SectionCard>

      {/* Audio (narration TTS) */}
      <SectionCard kicker="Audio Channel" title={t("default_audio_model")}>
        {audioBackends.length > 0 ? (
          <ProviderModelSelect
            value={currentAudioBackend}
            options={audioBackends}
            providerNames={allProviderNames}
            onChange={(v) => setDraft((prev) => ({ ...prev, default_audio_backend: v }))}
            allowDefault
            defaultLabel={t("auto_select")}
            defaultHint={t("auto")}
            aria-label={t("default_audio_model")}
          />
        ) : (
          emptyHint(t("no_audio_providers_hint"))
        )}

        <div className="mt-4 space-y-3.5">
          <div>
            <label
              htmlFor="narration-voice-input"
              className="mb-1.5 block font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-text-4"
            >
              {t("narration_voice_label")}
            </label>
            <input
              id="narration-voice-input"
              type="text"
              value={currentNarrationVoice}
              onChange={(e) => setDraft((prev) => ({ ...prev, narration_voice: e.target.value }))}
              className="w-full rounded-[8px] border border-hairline bg-bg-grad-a/55 px-3 py-2 text-[12.5px] text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            />
            <p className="mt-1 text-[11px] text-text-4">{t("narration_voice_hint")}</p>
          </div>
          <div>
            <label
              htmlFor="narration-speed-input"
              className="mb-1.5 block font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-text-4"
            >
              {t("narration_speed_label")}
            </label>
            <input
              id="narration-speed-input"
              type="number"
              min={0.1}
              step={0.1}
              value={currentNarrationSpeed ?? ""}
              onChange={(e) => {
                const raw = e.target.value;
                setDraft((prev) => {
                  if (raw === "") return { ...prev, narration_speed: null };
                  const next = Number(raw);
                  // 仅过滤非有限数：NaN/Infinity 会被 JSON 序列化为 null 误触"清除"语义。
                  // 0/负数允许临时存在（键入 0.5 会先经过 0），正数约束由保存时后端校验兜底。
                  if (!Number.isFinite(next)) return prev;
                  return { ...prev, narration_speed: next };
                });
              }}
              className="w-full rounded-[8px] border border-hairline bg-bg-grad-a/55 px-3 py-2 text-[12.5px] text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            />
            <p className="mt-1 text-[11px] text-text-4">{t("narration_speed_hint")}</p>
          </div>
        </div>
      </SectionCard>

      {/* Object storage (OSS) */}
      <SectionCard kicker="Object Storage" title={t("oss_title")} description={t("oss_desc")}>
        <div className="space-y-3.5">
          {ossField("oss_endpoint", t("oss_endpoint"), "oss-cn-hangzhou.aliyuncs.com")}
          {ossField("oss_bucket", t("oss_bucket"), "my-bucket")}
          {ossField("oss_access_key_id", t("oss_access_key_id"), "LTAI...")}
          <div>
            <div className="mb-1.5 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-text-4">
              {t("oss_access_key_secret")}
            </div>
            <input
              type="password"
              autoComplete="off"
              value={draft.oss_access_key_secret ?? ""}
              placeholder={
                settings.oss_access_key_secret.is_set
                  ? (settings.oss_access_key_secret.masked ?? "••••")
                  : t("oss_secret_placeholder")
              }
              onChange={(e) => setDraft((prev) => ({ ...prev, oss_access_key_secret: e.target.value }))}
              className={ossInputCls}
              aria-label={t("oss_access_key_secret")}
            />
          </div>
          {ossField("oss_upload_prefix", t("oss_upload_prefix"), "arcreel/")}
        </div>
      </SectionCard>

      {/* Provider asset cache (tecdo assetId) */}
      <SectionCard
        kicker="Asset Cache"
        title={t("asset_cache_title")}
        description={t("asset_cache_desc")}
      >
        <div className="space-y-4">
          <div>
            <label
              htmlFor="asset-cache-mode-select"
              className="mb-1.5 block font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-text-4"
            >
              {t("asset_cache_mode_label")}
            </label>
            <select
              id="asset-cache-mode-select"
              value={currentAssetCacheMode}
              onChange={(e) =>
                setDraft((prev) => ({
                  ...prev,
                  provider_asset_cache_mode: e.target.value as "cached" | "recreate",
                }))
              }
              className="w-full rounded-[8px] border border-hairline bg-bg-grad-a/55 px-3 py-2 text-[12.5px] text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              <option value="cached">{t("asset_cache_mode_cached")}</option>
              <option value="recreate">{t("asset_cache_mode_recreate")}</option>
            </select>
            <p className="mt-1 text-[11px] text-text-4">
              {currentAssetCacheMode === "recreate"
                ? t("asset_cache_mode_recreate_hint")
                : t("asset_cache_mode_cached_hint")}
            </p>
          </div>
          <div>
            <button
              type="button"
              onClick={() => void handleClearCache()}
              disabled={clearingCache}
              className="inline-flex items-center gap-2 rounded-[8px] border border-hairline bg-bg-grad-a/55 px-4 py-2 text-[12.5px] text-text-2 transition-colors hover:border-hairline-strong hover:bg-bg-grad-a hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-not-allowed disabled:opacity-60"
            >
              {clearingCache ? (
                <Loader2 className="h-3.5 w-3.5 motion-safe:animate-spin" aria-hidden />
              ) : null}
              {t("clear_asset_cache")}
            </button>
            <p className="mt-1 text-[11px] text-text-4">{t("clear_asset_cache_hint")}</p>
          </div>
        </div>
      </SectionCard>

      {/* Footer */}
      {isDirty && (
        <div className="flex gap-2 pt-1">
          <button
            type="button"
            onClick={() => void handleSave()}
            disabled={saving}
            className={ACCENT_BTN_CLS}
            style={ACCENT_BUTTON_STYLE}
          >
            {saving ? (
              <Loader2 className="h-3.5 w-3.5 motion-safe:animate-spin" aria-hidden />
            ) : null}
            {saving ? t("common:saving") : t("common:save")}
          </button>
          <button
            type="button"
            onClick={() => setDraft({})}
            className="rounded-[8px] border border-hairline bg-bg-grad-a/55 px-4 py-2 text-[12.5px] text-text-2 transition-colors hover:border-hairline-strong hover:bg-bg-grad-a hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            {t("common:reset")}
          </button>
        </div>
      )}
    </div>
  );
}
