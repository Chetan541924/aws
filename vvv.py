elif action_type == "RADIO":
    step_lower = step_name.lower()

    # Extract answer index (answer = 1 / 2)
    match = re.search(r"answer\s*=\s*(\d+)", step_lower)
    if not match:
        raise RuntimeError("RADIO step must specify answer = <number>")

    answer_index = match.group(1)

    # Resolve frames
    nav_frame, content_frame = resolve_ccs_frames(page)

    # ============================================================
    # PHASE 1: Wait for page to fully stabilize
    # ============================================================
    await page.wait_for_timeout(5000)  # Increased wait
    await content_frame.wait_for_load_state("domcontentloaded", timeout=15000)
    await page.wait_for_timeout(2000)  # Additional buffer

    # ============================================================
    # PHASE 2: Disable auto-scroll behavior
    # ============================================================
    await content_frame.evaluate("""
        () => {
            // Disable smooth scrolling
            document.documentElement.style.scrollBehavior = 'auto';
            
            // Prevent any scroll event listeners from firing
            window.addEventListener('scroll', (e) => {
                e.stopPropagation();
            }, true);
            
            // Lock scroll position temporarily
            const currentScroll = window.scrollY;
            window.scrollTo(0, currentScroll);
        }
    """)

    radio_selector = f"input[type='radio'][id$='answer{answer_index}']"
    
    # ============================================================
    # PHASE 3: Aggressive retry loop with scroll locking
    # ============================================================
    max_attempts = 5
    clicked = False
    
    for attempt in range(max_attempts):
        try:
            logger.info(
                LogCategory.EXECUTION,
                f"[PHASE 3] RADIO click attempt {attempt + 1}/{max_attempts}"
            )
            
            # Locate radio
            radio = content_frame.locator(radio_selector)
            
            if await radio.count() == 0:
                raise RuntimeError(f"Radio with id ending 'answer{answer_index}' not found")
            
            # Get element position
            box = await radio.first.bounding_box()
            if not box:
                logger.warning(LogCategory.EXECUTION, "Radio button has no bounding box, retrying...")
                await page.wait_for_timeout(1000)
                continue
            
            # ============================================================
            # Scroll and LOCK position
            # ============================================================
            await content_frame.evaluate(
                """(selector) => {
                    const element = document.querySelector(selector);
                    if (element) {
                        // Scroll to element
                        element.scrollIntoView({behavior: 'instant', block: 'center'});
                        
                        // LOCK the scroll position immediately
                        const lockedPosition = window.scrollY;
                        
                        // Prevent ANY scrolling for next 3 seconds
                        let scrollLockTimer = null;
                        const lockScroll = () => {
                            window.scrollTo(0, lockedPosition);
                        };
                        
                        window.addEventListener('scroll', lockScroll);
                        
                        scrollLockTimer = setTimeout(() => {
                            window.removeEventListener('scroll', lockScroll);
                        }, 3000);
                    }
                }""",
                radio_selector
            )
            
            # Wait for position to stabilize
            await page.wait_for_timeout(500)
            
            # Verify element is STILL visible after scroll lock
            await radio.first.wait_for(state="visible", timeout=5000)
            
            # ============================================================
            # Click using DIRECT JavaScript (bypasses all Playwright checks)
            # ============================================================
            click_result = await content_frame.evaluate(
                """(selector) => {
                    const element = document.querySelector(selector);
                    if (!element) {
                        return {success: false, error: 'Element not found'};
                    }
                    
                    // Check if visible
                    const rect = element.getBoundingClientRect();
                    const isVisible = rect.top >= 0 && rect.bottom <= window.innerHeight;
                    
                    if (!isVisible) {
                        return {success: false, error: 'Element not visible', top: rect.top, bottom: rect.bottom};
                    }
                    
                    // Force click
                    element.click();
                    
                    // Double-check selection
                    element.checked = true;
                    
                    // Trigger change event
                    const event = new Event('change', { bubbles: true });
                    element.dispatchEvent(event);
                    
                    return {success: true, checked: element.checked};
                }""",
                radio_selector
            )
            
            logger.info(
                LogCategory.EXECUTION,
                f"[PHASE 3] Click result: {click_result}"
            )
            
            if click_result.get('success'):
                # Wait and verify
                await page.wait_for_timeout(1000)
                
                is_checked = await radio.first.is_checked()
                if is_checked:
                    clicked = True
                    logger.info(
                        LogCategory.EXECUTION,
                        f"[PHASE 3] RADIO selected successfully: answer={answer_index}"
                    )
                    break
                else:
                    logger.warning(
                        LogCategory.EXECUTION,
                        "Radio clicked but not checked, retrying..."
                    )
            else:
                logger.warning(
                    LogCategory.EXECUTION,
                    f"Click failed: {click_result.get('error')}"
                )
                await page.wait_for_timeout(1000)
                
        except Exception as e:
            logger.warning(
                LogCategory.EXECUTION,
                f"[PHASE 3] Attempt {attempt + 1} failed: {str(e)}"
            )
            await page.wait_for_timeout(1000)
            
            if attempt == max_attempts - 1:
                # Last attempt - take screenshot for debugging
                await page.screenshot(path=f"radio_failure_{answer_index}.png")
    
    if not clicked:
        raise RuntimeError(
            f"Failed to click radio button answer={answer_index} after {max_attempts} attempts"
        )
    
    # Final wait for any page reactions
    await page.wait_for_timeout(2000)







# DEBUG: Monitor scroll events
await content_frame.evaluate("""
    () => {
        window.scrollEvents = [];
        window.addEventListener('scroll', () => {
            const stack = new Error().stack;
            window.scrollEvents.push({
                time: Date.now(),
                scrollY: window.scrollY,
                stack: stack
            });
        });
    }
""")

# ... your radio code here ...

# After failure, dump scroll events
try:
        scroll_events = await content_frame.evaluate("() => window.scrollEvents")
        logger.info(
            LogCategory.EXECUTION, 
            f"[DEBUG] Total scroll events captured: {len(scroll_events)}"
        )
        
        # Log each scroll event
        for idx, event in enumerate(scroll_events):
            logger.info(
                LogCategory.EXECUTION,
                f"[DEBUG] Scroll event {idx + 1}: "
                f"scrollY={event['scrollY']}, "
                f"time={event['time']}"
            )
            # Optionally log stack trace (can be verbose)
            # logger.info(LogCategory.EXECUTION, f"Stack: {event['stack']}")
            
    except Exception as e:
        logger.error(LogCategory.EXECUTION, f"[DEBUG] Failed to retrieve scroll events: {e}")
    
    # Final wait for any page reactions
    await page.wait_for_timeout(2000)
