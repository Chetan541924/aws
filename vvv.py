elif action_type == "CHECKBOX":

    import re

    step_lower = step_name.lower()

    # --------------------------------------------------
    # Resolve CCS frames
    # --------------------------------------------------
    nav_frame, content_frame = resolve_ccs_frames(page)

    # --------------------------------------------------
    # Locate ONLY real transaction rows
    # (Ignore headers, hidden rows, summary rows)
    # --------------------------------------------------
    rows = content_frame.locator(
        "xpath=//div[@id='ACCOUNT_ACTIVITY_Div']"
        "//tr[contains(@class,'subPanelContent')]"
    )

    row_count = await rows.count()
    if row_count == 0:
        raise RuntimeError("No transaction rows found in Account Activity")

    checkbox_to_click = None

    # ==================================================
    # CASE 1: ROW NUMBER (row 1 / row 2 / row N)
    # ==================================================
    row_match = re.search(r"row\s*(\d+)", step_lower)
    if row_match:
        row_index = int(row_match.group(1)) - 1

        if row_index < 0 or row_index >= row_count:
            raise RuntimeError(
                f"Row {row_index + 1} out of range (total rows: {row_count})"
            )

        row = rows.nth(row_index)

        checkbox_to_click = row.locator(
            "input[type='checkbox']:not([type='hidden'])"
        )

    # ==================================================
    # CASE 2: DESCRIPTION BASED
    # Example:
    # "checks the checkbox for transaction 'Zelle payment from Scott Snyder'"
    # ==================================================
    if checkbox_to_click is None:
        desc_match = re.search(r"transaction\s+'(.+?)'", step_name, re.IGNORECASE)
        if desc_match:
            description_text = desc_match.group(1).strip()

            for i in range(row_count):
                row = rows.nth(i)
                row_text = await row.inner_text()

                if description_text.lower() in row_text.lower():
                    checkbox_to_click = row.locator(
                        "input[type='checkbox']:not([type='hidden'])"
                    )
                    break

            if checkbox_to_click is None:
                raise RuntimeError(
                    f"Transaction with description '{description_text}' not found"
                )

    # ==================================================
    # FINAL VALIDATION
    # ==================================================
    if checkbox_to_click is None:
        raise RuntimeError(
            "CHECKBOX step must specify either row number or transaction description"
        )

    # --------------------------------------------------
    # Scroll + Wait + Check
    # --------------------------------------------------
    await checkbox_to_click.scroll_into_view_if_needed()
    await checkbox_to_click.wait_for(state="attached", timeout=15000)

    if not await checkbox_to_click.is_checked():
        await checkbox_to_click.check(force=True)

    logger.info(
        LogCategory.EXECUTION,
        "[PHASE 3] Checkbox selected in Account Activity"
    )
