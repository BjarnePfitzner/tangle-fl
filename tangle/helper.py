from enum import Enum

class TipSelectorIdentifiers(Enum):
    TIP_SELECTOR = 0
    ACCURACY_TIP_SELECTOR = 1
    MALICIOUS_TIP_SELECTOR = 2

def determineTipSelector(args, round_number):
    normal_start_rounds = args.tip_selector_from
    acc_start_rounds = args.acc_tip_selector_from
    mal_start_rounds = args.mal_tip_selector_from

    # Create a set of round numbers, to eliminate duplicates first
    normal_start_rounds = list(set(map(int, normal_start_rounds)))
    acc_start_rounds = list(set(map(int, acc_start_rounds)))
    mal_start_rounds = list(set(map(int, mal_start_rounds)))

    # Use TipSelector as default. One cannot specify 0 as default in argparser,
    # because append() action is used and users might specify 0 for a different tip selector
    if not any([True for x in (normal_start_rounds, acc_start_rounds, mal_start_rounds) if 0 in x]):
        normal_start_rounds.append(0)

    # Sort list ascending
    normal_start_rounds.sort()
    acc_start_rounds.sort()
    mal_start_rounds.sort()

    # For each possible tip selector, search for the closest round number,
    # that is bigger than other round start numbers, but <= the current round number
    closest_start_round_normal = next((x for x in reversed(normal_start_rounds) if x <= round_number), -1 * float("inf"))
    closest_start_round_acc = next((x for x in reversed(acc_start_rounds) if x <= round_number), -1 * float("inf"))
    closest_start_round_mal = next((x for x in reversed(mal_start_rounds) if x <= round_number), -1 * float("inf"))

    closest_start_round = max(closest_start_round_normal, closest_start_round_acc, closest_start_round_mal)
    if closest_start_round == closest_start_round_normal:
        return TipSelectorIdentifiers.TIP_SELECTOR
    elif closest_start_round == closest_start_round_acc:
        return TipSelectorIdentifiers.ACCURACY_TIP_SELECTOR
    elif closest_start_round == closest_start_round_mal:
        return TipSelectorIdentifiers.MALICIOUS_TIP_SELECTOR
    else:
        return TipSelectorIdentifiers.TIP_SELECTOR
