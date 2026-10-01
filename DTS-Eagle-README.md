# DTS AndroPort Project

We bring to you a collection of ports for DTS Audio's Android techonologies to run most Android devices. While most of these are in developement, some features are working and audible differences can be noticed between having the effect on and off. We have a team of skilled Android developers from different fields working on different parts of this mod to get the whole thing working. Some of the main contributers are:

  - Laster K. [@lazerl0rd](https://gitlab.com/lazerl0rd) (Android kernel developer)
  - JohnFawkes [@JohnFawkes](https://gitlab.com/JohnFawkes) (Core member of ICEPower port team)
  - Michi [@schneidersirsch.m](https://gitlab.com/schneidersirsch.m) (Smali editor for DTS ports)
  - Josh Choo [@joshuous](https://gitlab.com/joshuous) (Android kernel and ROM developer)
  - UltraM8 [@arssovetnikov](https://gitlab.com/arssovetnikov) (Founder of Ainur Sauron)
  - And others...

## DTS Headphone:X
### Requirements
* [x] You need to request your kernel developer to add this in
* [x] You should preferebly be running a custom ROM
* [x] You need to be on at least Android Nougat
* [x] A device running a Snapdragon chip with at least WCD9330 and a Hexagon DSP

### Kernel Commits
WCD9330
```
git fetch https://gitlab.com/dts_project/kernel -t hpx-wcd9330 && git cherry-pick fbc216a072055e4eb21f6838fcb7736da7b93c45 && git cherry-pick fbc216a072055e4eb21f6838fcb7736da7b93c45..bd142dc7db91d2afccfaf68887607bbd7b17e3cc
```
WCD9335
```
git fetch https://gitlab.com/dts_project/kernel -t hpx-wcd9335 &&  git cherry-pick a8d80fb256ee6fa7fa862a775c566e1f26f39122 && git cherry-pick a8d80fb256ee6fa7fa862a775c566e1f26f39122..037d6324aed1a1b10c22eee49f8be97eb735889f
```
WCD934X
```
git fetch https://gitlab.com/dts_project/kernel -t hpx-wcd934x && git cherry-pick a953da827af9a6a2e47fbca40bb77e42ed32f613 && git cherry-pick a953da827af9a6a2e47fbca40bb77e42ed32f613..1f8b2f615399729e1a090e8b8532536249394be4
```
...and for Oreo+ (on top of the commits set above)
```
git fetch https://gitlab.com/dts_project/kernel -t && git cherry-pick 391bb4b371eab92ea0ea66ad254df0abd7cf0353
```

If you're still having issues add the following to your defconfig:
```
CONFIG_DTS_SRS_TM=y
```
## DTS:X Ultra
### Requirements
* [x] You need to request your kernel developer to add this in
* [x] You should be on Oreo or higher

### Kernel Commits
Universal
```
git fetch https://gitlab.com/dts_project/kernel -t ultra && git cherry-pick c5b95ad1248761909a9d1891898b3cd35c377701 && git cherry-pick c5b95ad1248761909a9d1891898b3cd35c377701..229e94198375698866007e1dca6aff1821a80386
```
## Post-kernel implementation
Don't forget to tell your users to collect the userland side of the port (choose one) from https://forum.xda-developers.com/android/software/port-dtsx-ultra-dts-headphonex-t3896233!

## Contact Us
Join our Telegram Group at https://t.me/DTSWizardproxy/.
## Linux 4.19 + techpack/audio + WCD937x port (`rystx-re-dtsx-wcd937x`)

This tree carries a port of the DTS Headphone:X (HPX) kernel side to
Linux 4.19 (Qualcomm msm-4.19 downstream), integrated into the
`techpack/audio` layout (no files under legacy `sound/`) with support
for the WCD937x codec (bolero RX/TX macros + SoundWire + MBHC v2).

### Commits carried (branch `rystx-re-dtsx-wcd937x`)

| Commit | Content | Source |
|---|---|---|
| `b2268b38ddaf` | dts_eagle char driver import (`techpack/audio/dsp/dts_eagle/`, `dts_eagle_dlkm`) | hpx-wcd934x 113a2084817c + 391bb4b371ea (final upstream state) |
| `59493756531d` | DTS-Eagle DSP integration: `asoc/msm-dts-eagle.c` + `include/dsp/msm-dts-eagle.h`, `adm/q6asm/q6core` hooks, hwacc effects, compress/effects/routing wiring, `asm_dts_eagle_param` structs, `ADM_DTS_EAGLE` cal enum | hpx-wcd934x b71085cd3628 (adapted: component API, dma_buf ion API, include paths, EXPORT_SYMBOLs, Kbuild wiring) |
| `e5306f144f4e` | WCD937x + MBHC: `g_DebugMode`, debug-mode jack detection (`wcd_mbhc_plug_detect_for_debug_mode`), impedance capture (`g_ZL`/`g_ZR`), audiowizard extcon registration | hpx-wcd934x b2fd482177fb + 8576ddf2e618 + 82953807708b (adapted: wcd934x->wcd937x, switch_dev->extcon) |
| `f7a0a94ea60e` | AudioWizard calibration ioctls (`AUDIO_SET_AUDIOWIZARD_FORCE_PRESET`, `AUDIO_GET_HS_IMP`, `AUDIO_SET_MODE`, `AUDIO_SET_FORMAT`, compat variants) | hpx-wcd934x b2fd482177fb (adapted: 4.19 extcon API, mutex-safe get handler) |
| `16060ec7ca4c` | Enable `CONFIG_AINUR_DTS_HW=y` in `techpack/audio/config/fogauto.conf` | hpx-wcd934x 1f8b2f615399 (adapted to techpack fragment) |

Skipped upstream commits: a953da827af9 (removal of a legacy Kconfig; nothing
to remove here) and the unused ASUS `regmap/internal.h` struct additions
(nothing references them).

### Required defconfig options

In `techpack/audio/config/<device>auto.conf` (fog uses `fogauto.conf`):

    CONFIG_AINUR_DTS_HW=y

No mainline defconfig changes are required. The legacy README hint
`CONFIG_DTS_SRS_TM=y` has no effect in this tree (it is not a Kconfig
symbol here); the SRS_TM module stays compiled out.

### Userland-visible interface

Unchanged from the upstream HPX port:

- `/dev/dts_eagle1..6` char device (dts_eagle_dlkm) for param caches,
  licenses and device state.
- hwdep ioctls `DTS_EAGLE_IOCTL_*` (0xF2 driver id) on the BE hwdep node.
- "DTS HPX" ALSA mixer controls on the msm-pcm-routing component.

Deviations (documented, kept compatible where possible):

- The ASUS `audiowizard_force_preset` indicator is an extcon device
  (cable `EXTCON_JACK_HEADPHONE`) registered by the wcd937x codec
  instead of the legacy `switch_dev`; the 4.19 extcon core derives its
  name from the codec device. Controlled via the
  `AUDIO_SET_AUDIOWIZARD_FORCE_PRESET` calibration ioctl.
- The AudioWizard cal types (AUDIOWIZARD_FORCE_PRESET, SKYPE_STATE,
  SET_MODE, SET_FORMAT, GET_IMP) shift the cal-type enum values of
  everything after `ADM_RTAC_AUDVOL_CAL_TYPE` by 5 (ASUS ABI contract).
- `wcd_mbhc_plug_detect_for_debug_mode()` and `g_DebugMode` are exported
  but nothing in-kernel calls them (same as upstream); with
  `g_DebugMode = 0` normal MBHC headset detection runs.

### Known limitations

- Not verified on hardware (no audible/on-device claims). Compile-verified
  for arm64 with and without `CONFIG_AINUR_DTS_HW`; module link/modpost
  runs only in a full `make modules`, which was not performed here.
- The eagle out-of-band memory path assumes the DTS HPX ADM topology is
  selected by ACDB/userland; with other topologies nothing changes.
- HPX effect + SA+ coexistence follows upstream: HPX on suppresses SA+
  effect params while active.
