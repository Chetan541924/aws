elif action_type == "RADIO":
    step_lower = step_name.lower()

    # --------------------------------------------------
    # 1. Extract answer index
    # --------------------------------------------------
    match = re.search(r"answer\s*=\s*(\d+)", step_lower)
    if not match:
        raise RuntimeError("RADIO step must specify answer = <number>")

    answer_index = int(match.group(1))

    # --------------------------------------------------
    # 2. Resolve frames fresh
    # --------------------------------------------------
    nav_frame, content_frame = resolve_ccs_frames(page)

    # --------------------------------------------------
    # 3. Discover radio groups (read-only)
    # --------------------------------------------------
    radio_groups = await content_frame.evaluate("""
        () => {
            const radios = Array.from(document.querySelectorAll('input[type="radio"]'));
            const groups = {};
            radios.forEach(r => {
                if (!groups[r.name]) groups[r.name] = [];
                if (r.id && r.offsetParent !== null) {
                    groups[r.name].push({ id: r.id, value: r.value });
                }
            });
            return groups;
        }
    """)

    if not radio_groups:
        raise RuntimeError("No radio buttons found on page")

    # --------------------------------------------------
    # 4. Select target group
    # --------------------------------------------------
    target_group = None
    for name, radios in radio_groups.items():
        if "answer" in name.lower() and len(radios) >= 2:
            target_group = radios
            break

    if not target_group:
        raise RuntimeError("No valid radio group found")

    idx = answer_index - 1
    if idx < 0 or idx >= len(target_group):
        raise RuntimeError("Radio index out of range")

    radio_id = target_group[idx]["id"]

    # --------------------------------------------------
    # 5. Locate radio BEFORE click
    # --------------------------------------------------
    radio = content_frame.locator(f"#{radio_id}")
    await radio.wait_for(state="visible", timeout=10000)

    # --------------------------------------------------
    # 6. CLICK (triggers JSF partial submit)
    # --------------------------------------------------
    await radio.click(force=True)

    # --------------------------------------------------
    # 🔥 7. WAIT FOR JSF PARTIAL REFRESH (MANDATORY)
    # --------------------------------------------------
    await content_frame.wait_for_selector(
        f"#{radio_id}", state="detached", timeout=15000
    )
    await content_frame.wait_for_selector(
        f"#{radio_id}", state="attached", timeout=15000
    )

    # --------------------------------------------------
    # 8. Re-resolve frames AFTER refresh
    # --------------------------------------------------
    nav_frame, content_frame = resolve_ccs_frames(page)

    # --------------------------------------------------
    # 9. Re-locate radio in NEW DOM
    # --------------------------------------------------
    radio_new = content_frame.locator(f"#{radio_id}")
    await radio_new.wait_for(state="visible", timeout=10000)

    # --------------------------------------------------
    # 10. Verify selection
    # --------------------------------------------------
    if not await radio_new.is_checked():
        raise RuntimeError("Radio click did not persist after JSF refresh")

    logger.info(
        LogCategory.EXECUTION,
        f"[PHASE 3] RADIO selected successfully: answer={answer_index}, id={radio_id}"
    )
