import { ChargeService } from "./service";

export class CheckoutFlow extends BaseFlow {
    private sessionId: string;
    async submit(amount: number): Promise<boolean> { return true; }
}

export function formatAmount(cents: number): string { return ""; }
