/**
 * @license
 * Copyright Google Inc. All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.io/license
 */
(function (factory) {
    if (typeof module === "object" && typeof module.exports === "object") {
        var v = factory(require, exports);
        if (v !== undefined) module.exports = v;
    }
    else if (typeof define === "function" && define.amd) {
        define("@angular/language-service/src/types", ["require", "exports", "@angular/compiler-cli/src/language_services"], factory);
    }
})(function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    var language_services_1 = require("@angular/compiler-cli/src/language_services");
    exports.BuiltinType = language_services_1.BuiltinType;
    /**
     * The kind of diagnostic message.
     *
     * @publicApi
     */
    var DiagnosticKind;
    (function (DiagnosticKind) {
        DiagnosticKind[DiagnosticKind["Error"] = 0] = "Error";
        DiagnosticKind[DiagnosticKind["Warning"] = 1] = "Warning";
    })(DiagnosticKind = exports.DiagnosticKind || (exports.DiagnosticKind = {}));
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHlwZXMuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi9wYWNrYWdlcy9sYW5ndWFnZS1zZXJ2aWNlL3NyYy90eXBlcy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7O0dBTUc7Ozs7Ozs7Ozs7OztJQUdILGlGQUE0TDtJQUkxTCxzQkFKTSwrQkFBVyxDQUlOO0lBb1BiOzs7O09BSUc7SUFDSCxJQUFZLGNBR1g7SUFIRCxXQUFZLGNBQWM7UUFDeEIscURBQUssQ0FBQTtRQUNMLHlEQUFPLENBQUE7SUFDVCxDQUFDLEVBSFcsY0FBYyxHQUFkLHNCQUFjLEtBQWQsc0JBQWMsUUFHekIiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGUgbGljZW5zZSB0aGF0IGNhbiBiZVxuICogZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBhdCBodHRwczovL2FuZ3VsYXIuaW8vbGljZW5zZVxuICovXG5cbmltcG9ydCB7Q29tcGlsZURpcmVjdGl2ZU1ldGFkYXRhLCBDb21waWxlTWV0YWRhdGFSZXNvbHZlciwgQ29tcGlsZVBpcGVTdW1tYXJ5LCBOZ0FuYWx5emVkTW9kdWxlcywgU3RhdGljU3ltYm9sfSBmcm9tICdAYW5ndWxhci9jb21waWxlcic7XG5pbXBvcnQge0J1aWx0aW5UeXBlLCBEZWNsYXJhdGlvbktpbmQsIERlZmluaXRpb24sIFBpcGVJbmZvLCBQaXBlcywgU2lnbmF0dXJlLCBTcGFuLCBTeW1ib2wsIFN5bWJvbERlY2xhcmF0aW9uLCBTeW1ib2xRdWVyeSwgU3ltYm9sVGFibGV9IGZyb20gJ0Bhbmd1bGFyL2NvbXBpbGVyLWNsaS9zcmMvbGFuZ3VhZ2Vfc2VydmljZXMnO1xuaW1wb3J0IHtBc3RSZXN1bHQsIFRlbXBsYXRlSW5mb30gZnJvbSAnLi9jb21tb24nO1xuXG5leHBvcnQge1xuICBCdWlsdGluVHlwZSxcbiAgRGVjbGFyYXRpb25LaW5kLFxuICBEZWZpbml0aW9uLFxuICBQaXBlSW5mbyxcbiAgUGlwZXMsXG4gIFNpZ25hdHVyZSxcbiAgU3BhbixcbiAgU3ltYm9sLFxuICBTeW1ib2xEZWNsYXJhdGlvbixcbiAgU3ltYm9sUXVlcnksXG4gIFN5bWJvbFRhYmxlXG59O1xuXG4vKipcbiAqIFRoZSBpbmZvcm1hdGlvbiBgTGFuZ3VhZ2VTZXJ2aWNlYCBuZWVkcyBmcm9tIHRoZSBgTGFuZ3VhZ2VTZXJ2aWNlSG9zdGAgdG8gZGVzY3JpYmUgdGhlIGNvbnRlbnQgb2ZcbiAqIGEgdGVtcGxhdGUgYW5kIHRoZSBsYW5ndWFnZSBjb250ZXh0IHRoZSB0ZW1wbGF0ZSBpcyBpbi5cbiAqXG4gKiBBIGhvc3QgaW50ZXJmYWNlOyBzZWUgYExhbmd1YWdlU2VydmljZUhvc3RgLlxuICpcbiAqIEBwdWJsaWNBcGlcbiAqL1xuZXhwb3J0IGludGVyZmFjZSBUZW1wbGF0ZVNvdXJjZSB7XG4gIC8qKlxuICAgKiBUaGUgc291cmNlIG9mIHRoZSB0ZW1wbGF0ZS5cbiAgICovXG4gIHJlYWRvbmx5IHNvdXJjZTogc3RyaW5nO1xuXG4gIC8qKlxuICAgKiBUaGUgdmVyc2lvbiBvZiB0aGUgc291cmNlLiBBcyBmaWxlcyBhcmUgbW9kaWZpZWQgdGhlIHZlcnNpb24gc2hvdWxkIGNoYW5nZS4gVGhhdCBpcywgaWYgdGhlXG4gICAqIGBMYW5ndWFnZVNlcnZpY2VgIHJlcXVlc3RpbmcgdGVtcGxhdGUgaW5mb3JtYXRpb24gZm9yIGEgc291cmNlIGZpbGUgYW5kIHRoYXQgZmlsZSBoYXMgY2hhbmdlZFxuICAgKiBzaW5jZSB0aGUgbGFzdCB0aW1lIHRoZSBob3N0IHdhcyBhc2tlZCBmb3IgdGhlIGZpbGUgdGhlbiB0aGlzIHZlcnNpb24gc3RyaW5nIHNob3VsZCBiZVxuICAgKiBkaWZmZXJlbnQuIE5vIGFzc3VtcHRpb25zIGFyZSBtYWRlIGFib3V0IHRoZSBmb3JtYXQgb2YgdGhpcyBzdHJpbmcuXG4gICAqXG4gICAqIFRoZSB2ZXJzaW9uIGNhbiBjaGFuZ2UgbW9yZSBvZnRlbiB0aGFuIHRoZSBzb3VyY2UgYnV0IHNob3VsZCBub3QgY2hhbmdlIGxlc3Mgb2Z0ZW4uXG4gICAqL1xuICByZWFkb25seSB2ZXJzaW9uOiBzdHJpbmc7XG5cbiAgLyoqXG4gICAqIFRoZSBzcGFuIG9mIHRoZSB0ZW1wbGF0ZSB3aXRoaW4gdGhlIHNvdXJjZSBmaWxlLlxuICAgKi9cbiAgcmVhZG9ubHkgc3BhbjogU3BhbjtcblxuICAvKipcbiAgICogQSBzdGF0aWMgc3ltYm9sIGZvciB0aGUgdGVtcGxhdGUncyBjb21wb25lbnQuXG4gICAqL1xuICByZWFkb25seSB0eXBlOiBTdGF0aWNTeW1ib2w7XG5cbiAgLyoqXG4gICAqIFRoZSBgU3ltYm9sVGFibGVgIGZvciB0aGUgbWVtYmVycyBvZiB0aGUgY29tcG9uZW50LlxuICAgKi9cbiAgcmVhZG9ubHkgbWVtYmVyczogU3ltYm9sVGFibGU7XG5cbiAgLyoqXG4gICAqIEEgYFN5bWJvbFF1ZXJ5YCBmb3IgdGhlIGNvbnRleHQgb2YgdGhlIHRlbXBsYXRlLlxuICAgKi9cbiAgcmVhZG9ubHkgcXVlcnk6IFN5bWJvbFF1ZXJ5O1xufVxuXG4vKipcbiAqIEEgc2VxdWVuY2Ugb2YgdGVtcGxhdGUgc291cmNlcy5cbiAqXG4gKiBBIGhvc3QgdHlwZTsgc2VlIGBMYW5ndWFnZVNlcnZpY2VIb3N0YC5cbiAqXG4gKiBAcHVibGljQXBpXG4gKi9cbmV4cG9ydCB0eXBlIFRlbXBsYXRlU291cmNlcyA9IFRlbXBsYXRlU291cmNlW10gfCB1bmRlZmluZWQ7XG5cblxuLyoqXG4gKiBFcnJvciBpbmZvcm1hdGlvbiBmb3VuZCBnZXR0aW5nIGRlY2xhcmF0aW9uIGluZm9ybWF0aW9uXG4gKlxuICogQSBob3N0IHR5cGU7IHNlZSBgTGFuZ3VhZ2VTZXJ2aWNlSG9zdGAuXG4gKlxuICogQHB1YmxpY0FwaVxuICovXG5leHBvcnQgaW50ZXJmYWNlIERlY2xhcmF0aW9uRXJyb3Ige1xuICAvKipcbiAgICogVGhlIHNwYW4gb2YgdGhlIGVycm9yIGluIHRoZSBkZWNsYXJhdGlvbidzIG1vZHVsZS5cbiAgICovXG4gIHJlYWRvbmx5IHNwYW46IFNwYW47XG5cbiAgLyoqXG4gICAqIFRoZSBtZXNzYWdlIHRvIGRpc3BsYXkgZGVzY3JpYmluZyB0aGUgZXJyb3Igb3IgYSBjaGFpblxuICAgKiBvZiBtZXNzYWdlcy5cbiAgICovXG4gIHJlYWRvbmx5IG1lc3NhZ2U6IHN0cmluZ3xEaWFnbm9zdGljTWVzc2FnZUNoYWluO1xufVxuXG4vKipcbiAqIEluZm9ybWF0aW9uIGFib3V0IHRoZSBjb21wb25lbnQgZGVjbGFyYXRpb25zLlxuICpcbiAqIEEgZmlsZSBtaWdodCBjb250YWluIGEgZGVjbGFyYXRpb24gd2l0aG91dCBhIHRlbXBsYXRlIGJlY2F1c2UgdGhlIGZpbGUgY29udGFpbnMgb25seVxuICogdGVtcGxhdGVVcmwgcmVmZXJlbmNlcy4gSG93ZXZlciwgdGhlIGNvbXBvbmVudCBkZWNsYXJhdGlvbiBtaWdodCBjb250YWluIGVycm9ycyB0aGF0XG4gKiBuZWVkIHRvIGJlIHJlcG9ydGVkIHN1Y2ggYXMgdGhlIHRlbXBsYXRlIHN0cmluZyBpcyBtaXNzaW5nIG9yIHRoZSBjb21wb25lbnQgaXMgbm90XG4gKiBkZWNsYXJlZCBpbiBhIG1vZHVsZS4gVGhlc2UgZXJyb3Igc2hvdWxkIGJlIHJlcG9ydGVkIG9uIHRoZSBkZWNsYXJhdGlvbiwgbm90IHRoZVxuICogdGVtcGxhdGUuXG4gKlxuICogQSBob3N0IHR5cGU7IHNlZSBgTGFuZ3VhZ2VTZXJ2aWNlSG9zdGAuXG4gKlxuICogQHB1YmxpY0FwaVxuICovXG5leHBvcnQgaW50ZXJmYWNlIERlY2xhcmF0aW9uIHtcbiAgLyoqXG4gICAqIFRoZSBzdGF0aWMgc3ltYm9sIG9mIHRoZSBjb21wcG9uZW50IGJlaW5nIGRlY2xhcmVkLlxuICAgKi9cbiAgcmVhZG9ubHkgdHlwZTogU3RhdGljU3ltYm9sO1xuXG4gIC8qKlxuICAgKiBUaGUgc3BhbiBvZiB0aGUgZGVjbGFyYXRpb24gYW5ub3RhdGlvbiByZWZlcmVuY2UgKGUuZy4gdGhlICdDb21wb25lbnQnIG9yICdEaXJlY3RpdmUnXG4gICAqIHJlZmVyZW5jZSkuXG4gICAqL1xuICByZWFkb25seSBkZWNsYXJhdGlvblNwYW46IFNwYW47XG5cbiAgLyoqXG4gICAqIFJlZmVyZW5jZSB0byB0aGUgY29tcGlsZXIgZGlyZWN0aXZlIG1ldGFkYXRhIGZvciB0aGUgZGVjbGFyYXRpb24uXG4gICAqL1xuICByZWFkb25seSBtZXRhZGF0YT86IENvbXBpbGVEaXJlY3RpdmVNZXRhZGF0YTtcblxuICAvKipcbiAgICogRXJyb3IgcmVwb3J0ZWQgdHJ5aW5nIHRvIGdldCB0aGUgbWV0YWRhdGEuXG4gICAqL1xuICByZWFkb25seSBlcnJvcnM6IERlY2xhcmF0aW9uRXJyb3JbXTtcbn1cblxuLyoqXG4gKiBBIHNlcXVlbmNlIG9mIGRlY2xhcmF0aW9ucy5cbiAqXG4gKiBBIGhvc3QgdHlwZTsgc2VlIGBMYW5ndWFnZVNlcnZpY2VIb3N0YC5cbiAqXG4gKiBAcHVibGljQXBpXG4gKi9cbmV4cG9ydCB0eXBlIERlY2xhcmF0aW9ucyA9IERlY2xhcmF0aW9uW107XG5cbi8qKlxuICogVGhlIGhvc3QgZm9yIGEgYExhbmd1YWdlU2VydmljZWAuIFRoaXMgcHJvdmlkZXMgYWxsIHRoZSBgTGFuZ3VhZ2VTZXJ2aWNlYCByZXF1aXJlcyB0byByZXNwb25kXG4gKiB0byB0aGUgYExhbmd1YWdlU2VydmljZWAgcmVxdWVzdHMuXG4gKlxuICogVGhpcyBpbnRlcmZhY2UgZGVzY3JpYmVzIHRoZSByZXF1aXJlbWVudHMgb2YgdGhlIGBMYW5ndWFnZVNlcnZpY2VgIG9uIGl0cyBob3N0LlxuICpcbiAqIFRoZSBob3N0IGludGVyZmFjZSBpcyBob3N0IGxhbmd1YWdlIGFnbm9zdGljLlxuICpcbiAqIEFkZGluZyBvcHRpb25hbCBtZW1iZXIgdG8gdGhpcyBpbnRlcmZhY2Ugb3IgYW55IGludGVyZmFjZSB0aGF0IGlzIGRlc2NyaWJlZCBhcyBhXG4gKiBgTGFuZ3VhZ2VTZXJ2aWNlSG9zdGAgaW50ZXJmYWNlIGlzIG5vdCBjb25zaWRlcmVkIGEgYnJlYWtpbmcgY2hhbmdlIGFzIGRlZmluZWQgYnkgU2VtVmVyLlxuICogUmVtb3ZpbmcgYSBtZXRob2Qgb3IgY2hhbmdpbmcgYSBtZW1iZXIgZnJvbSByZXF1aXJlZCB0byBvcHRpb25hbCB3aWxsIGFsc28gbm90IGJlIGNvbnNpZGVyZWQgYVxuICogYnJlYWtpbmcgY2hhbmdlLlxuICpcbiAqIElmIGEgbWVtYmVyIGlzIGRlcHJlY2F0ZWQgaXQgd2lsbCBiZSBjaGFuZ2VkIHRvIG9wdGlvbmFsIGluIGEgbWlub3IgcmVsZWFzZSBiZWZvcmUgaXQgaXNcbiAqIHJlbW92ZWQgaW4gYSBtYWpvciByZWxlYXNlLlxuICpcbiAqIEFkZGluZyBhIHJlcXVpcmVkIG1lbWJlciBvciBjaGFuZ2luZyBhIG1ldGhvZCdzIHBhcmFtZXRlcnMsIGlzIGNvbnNpZGVyZWQgYSBicmVha2luZyBjaGFuZ2UgYW5kXG4gKiB3aWxsIG9ubHkgYmUgZG9uZSB3aGVuIGJyZWFraW5nIGNoYW5nZXMgYXJlIGFsbG93ZWQuIFdoZW4gcG9zc2libGUsIGEgbmV3IG9wdGlvbmFsIG1lbWJlciB3aWxsXG4gKiBiZSBhZGRlZCBhbmQgdGhlIG9sZCBtZW1iZXIgd2lsbCBiZSBkZXByZWNhdGVkLiBUaGUgbmV3IG1lbWJlciB3aWxsIHRoZW4gYmUgbWFkZSByZXF1aXJlZCBpblxuICogYW5kIHRoZSBvbGQgbWVtYmVyIHdpbGwgYmUgcmVtb3ZlZCBvbmx5IHdoZW4gYnJlYWtpbmcgY2hhbmdlcyBhcmUgYWxsb3dlZC5cbiAqXG4gKiBXaGlsZSBhbiBpbnRlcmZhY2UgaXMgbWFya2VkIGFzIGV4cGVyaW1lbnRhbCBicmVha2luZy1jaGFuZ2VzIHdpbGwgYmUgYWxsb3dlZCBiZXR3ZWVuIG1pbm9yXG4gKiByZWxlYXNlcy4gQWZ0ZXIgYW4gaW50ZXJmYWNlIGlzIG1hcmtlZCBhcyBzdGFibGUgYnJlYWtpbmctY2hhbmdlcyB3aWxsIG9ubHkgYmUgYWxsb3dlZCBiZXR3ZWVuXG4gKiBtYWpvciByZWxlYXNlcy4gTm8gYnJlYWtpbmcgY2hhbmdlcyBhcmUgYWxsb3dlZCBiZXR3ZWVuIHBhdGNoIHJlbGVhc2VzLlxuICpcbiAqIEBwdWJsaWNBcGlcbiAqL1xuZXhwb3J0IGludGVyZmFjZSBMYW5ndWFnZVNlcnZpY2VIb3N0IHtcbiAgLyoqXG4gICAqIFRoZSByZXNvbHZlciB0byB1c2UgdG8gZmluZCBjb21waWxlciBtZXRhZGF0YS5cbiAgICovXG4gIHJlYWRvbmx5IHJlc29sdmVyOiBDb21waWxlTWV0YWRhdGFSZXNvbHZlcjtcblxuICAvKipcbiAgICogUmV0dXJucyB0aGUgdGVtcGxhdGUgaW5mb3JtYXRpb24gZm9yIHRlbXBsYXRlcyBpbiBgZmlsZU5hbWVgIGF0IHRoZSBnaXZlbiBsb2NhdGlvbi4gSWZcbiAgICogYGZpbGVOYW1lYCByZWZlcnMgdG8gYSB0ZW1wbGF0ZSBmaWxlIHRoZW4gdGhlIGBwb3NpdGlvbmAgc2hvdWxkIGJlIGlnbm9yZWQuIElmIHRoZSBgcG9zaXRpb25gXG4gICAqIGlzIG5vdCBpbiBhIHRlbXBsYXRlIGxpdGVyYWwgc3RyaW5nIHRoZW4gdGhpcyBtZXRob2Qgc2hvdWxkIHJldHVybiBgdW5kZWZpbmVkYC5cbiAgICovXG4gIGdldFRlbXBsYXRlQXQoZmlsZU5hbWU6IHN0cmluZywgcG9zaXRpb246IG51bWJlcik6IFRlbXBsYXRlU291cmNlfHVuZGVmaW5lZDtcblxuICAvKipcbiAgICogUmV0dXJuIHRoZSB0ZW1wbGF0ZSBzb3VyY2UgaW5mb3JtYXRpb24gZm9yIGFsbCB0ZW1wbGF0ZXMgaW4gYGZpbGVOYW1lYCBvciBmb3IgYGZpbGVOYW1lYCBpZlxuICAgKiBpdCBpcyBhIHRlbXBsYXRlIGZpbGUuXG4gICAqL1xuICBnZXRUZW1wbGF0ZXMoZmlsZU5hbWU6IHN0cmluZyk6IFRlbXBsYXRlU291cmNlcztcblxuICAvKipcbiAgICogUmV0dXJucyB0aGUgQW5ndWxhciBkZWNsYXJhdGlvbnMgaW4gdGhlIGdpdmVuIGZpbGUuXG4gICAqL1xuICBnZXREZWNsYXJhdGlvbnMoZmlsZU5hbWU6IHN0cmluZyk6IERlY2xhcmF0aW9ucztcblxuICAvKipcbiAgICogUmV0dXJuIGEgc3VtbWFyeSBvZiBhbGwgQW5ndWxhciBtb2R1bGVzIGluIHRoZSBwcm9qZWN0LlxuICAgKi9cbiAgZ2V0QW5hbHl6ZWRNb2R1bGVzKCk6IE5nQW5hbHl6ZWRNb2R1bGVzO1xuXG4gIC8qKlxuICAgKiBSZXR1cm4gYSBsaXN0IGFsbCB0aGUgdGVtcGxhdGUgZmlsZXMgcmVmZXJlbmNlZCBieSB0aGUgcHJvamVjdC5cbiAgICovXG4gIGdldFRlbXBsYXRlUmVmZXJlbmNlcygpOiBzdHJpbmdbXTtcblxuICAvKipcbiAgICogUmV0dXJuIHRoZSBBU1QgZm9yIGJvdGggSFRNTCBhbmQgdGVtcGxhdGUgZm9yIHRoZSBjb250ZXh0RmlsZS5cbiAgICovXG4gIGdldFRlbXBsYXRlQXN0KHRlbXBsYXRlOiBUZW1wbGF0ZVNvdXJjZSwgY29udGV4dEZpbGU6IHN0cmluZyk6IEFzdFJlc3VsdDtcblxuICAvKipcbiAgICogUmV0dXJuIHRoZSB0ZW1wbGF0ZSBBU1QgZm9yIHRoZSBub2RlIHRoYXQgY29ycmVzcG9uZHMgdG8gdGhlIHBvc2l0aW9uLlxuICAgKi9cbiAgZ2V0VGVtcGxhdGVBc3RBdFBvc2l0aW9uKGZpbGVOYW1lOiBzdHJpbmcsIHBvc2l0aW9uOiBudW1iZXIpOiBUZW1wbGF0ZUluZm98dW5kZWZpbmVkO1xufVxuXG4vKipcbiAqIEFuIGl0ZW0gb2YgdGhlIGNvbXBsZXRpb24gcmVzdWx0IHRvIGJlIGRpc3BsYXllZCBieSBhbiBlZGl0b3IuXG4gKlxuICogQSBgTGFuZ3VhZ2VTZXJ2aWNlYCBpbnRlcmZhY2UuXG4gKlxuICogQHB1YmxpY0FwaVxuICovXG5leHBvcnQgaW50ZXJmYWNlIENvbXBsZXRpb24ge1xuICAvKipcbiAgICogVGhlIGtpbmQgb2YgY29tcGxldGlvbi5cbiAgICovXG4gIGtpbmQ6IERlY2xhcmF0aW9uS2luZDtcblxuICAvKipcbiAgICogVGhlIG5hbWUgb2YgdGhlIGNvbXBsZXRpb24gdG8gYmUgZGlzcGxheWVkXG4gICAqL1xuICBuYW1lOiBzdHJpbmc7XG5cbiAgLyoqXG4gICAqIFRoZSBrZXkgdG8gdXNlIHRvIHNvcnQgdGhlIGNvbXBsZXRpb25zIGZvciBkaXNwbGF5LlxuICAgKi9cbiAgc29ydDogc3RyaW5nO1xufVxuXG4vKipcbiAqIEEgc2VxdWVuY2Ugb2YgY29tcGxldGlvbnMuXG4gKlxuICogQHB1YmxpY0FwaVxuICovXG5leHBvcnQgdHlwZSBDb21wbGV0aW9ucyA9IENvbXBsZXRpb25bXSB8IHVuZGVmaW5lZDtcblxuLyoqXG4gKiBBIGZpbGUgYW5kIHNwYW4uXG4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgTG9jYXRpb24ge1xuICBmaWxlTmFtZTogc3RyaW5nO1xuICBzcGFuOiBTcGFuO1xufVxuXG4vKipcbiAqIFRoZSBraW5kIG9mIGRpYWdub3N0aWMgbWVzc2FnZS5cbiAqXG4gKiBAcHVibGljQXBpXG4gKi9cbmV4cG9ydCBlbnVtIERpYWdub3N0aWNLaW5kIHtcbiAgRXJyb3IsXG4gIFdhcm5pbmcsXG59XG5cbi8qKlxuICogQSB0ZW1wbGF0ZSBkaWFnbm9zdGljcyBtZXNzYWdlIGNoYWluLiBUaGlzIGlzIHNpbWlsYXIgdG8gdGhlIFR5cGVTY3JpcHRcbiAqIERpYWdub3N0aWNNZXNzYWdlQ2hhaW4uIFRoZSBtZXNzYWdlcyBhcmUgaW50ZW5kZWQgdG8gYmUgZm9ybWF0dGVkIGFzIHNlcGFyYXRlXG4gKiBzZW50ZW5jZSBmcmFnbWVudHMgYW5kIGluZGVudGVkLlxuICpcbiAqIEZvciBjb21wYXRpYmlsaXR5IHByZXZpb3VzIGltcGxlbWVudGF0aW9uLCB0aGUgdmFsdWVzIGFyZSBleHBlY3RlZCB0byBvdmVycmlkZVxuICogdG9TdHJpbmcoKSB0byByZXR1cm4gYSBmb3JtYXR0ZWQgbWVzc2FnZS5cbiAqXG4gKiBAcHVibGljQXBpXG4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgRGlhZ25vc3RpY01lc3NhZ2VDaGFpbiB7XG4gIC8qKlxuICAgKiBUaGUgdGV4dCBvZiB0aGUgZGlhZ25vc3RpYyBtZXNzYWdlIHRvIGRpc3BsYXkuXG4gICAqL1xuICBtZXNzYWdlOiBzdHJpbmc7XG5cbiAgLyoqXG4gICAqIFRoZSBuZXh0IG1lc3NhZ2UgaW4gdGhlIGNoYWluLlxuICAgKi9cbiAgbmV4dD86IERpYWdub3N0aWNNZXNzYWdlQ2hhaW47XG59XG5cbi8qKlxuICogQW4gdGVtcGxhdGUgZGlhZ25vc3RpYyBtZXNzYWdlIHRvIGRpc3BsYXkuXG4gKlxuICogQHB1YmxpY0FwaVxuICovXG5leHBvcnQgaW50ZXJmYWNlIERpYWdub3N0aWMge1xuICAvKipcbiAgICogVGhlIGtpbmQgb2YgZGlhZ25vc3RpYyBtZXNzYWdlXG4gICAqL1xuICBraW5kOiBEaWFnbm9zdGljS2luZDtcblxuICAvKipcbiAgICogVGhlIHNvdXJjZSBzcGFuIHRoYXQgc2hvdWxkIGJlIGhpZ2hsaWdodGVkLlxuICAgKi9cbiAgc3BhbjogU3BhbjtcblxuICAvKipcbiAgICogVGhlIHRleHQgb2YgdGhlIGRpYWdub3N0aWMgbWVzc2FnZSB0byBkaXNwbGF5IG9yIGEgY2hhaW4gb2YgbWVzc2FnZXMuXG4gICAqL1xuICBtZXNzYWdlOiBzdHJpbmd8RGlhZ25vc3RpY01lc3NhZ2VDaGFpbjtcbn1cblxuLyoqXG4gKiBBIHNlcXVlbmNlIG9mIGRpYWdub3N0aWMgbWVzc2FnZS5cbiAqXG4gKiBAcHVibGljQXBpXG4gKi9cbmV4cG9ydCB0eXBlIERpYWdub3N0aWNzID0gRGlhZ25vc3RpY1tdO1xuXG4vKipcbiAqIEEgc2VjdGlvbiBvZiBob3ZlciB0ZXh0LiBJZiB0aGUgdGV4dCBpcyBjb2RlIHRoZW4gbGFuZ3VhZ2Ugc2hvdWxkIGJlIHByb3ZpZGVkLlxuICogT3RoZXJ3aXNlIHRoZSB0ZXh0IGlzIGFzc3VtZWQgdG8gYmUgTWFya2Rvd24gdGV4dCB0aGF0IHdpbGwgYmUgc2FuaXRpemVkLlxuICovXG5leHBvcnQgaW50ZXJmYWNlIEhvdmVyVGV4dFNlY3Rpb24ge1xuICAvKipcbiAgICogU291cmNlIGNvZGUgb3IgbWFya2Rvd24gdGV4dCBkZXNjcmliaW5nIHRoZSBzeW1ib2wgYSB0aGUgaG92ZXIgbG9jYXRpb24uXG4gICAqL1xuICByZWFkb25seSB0ZXh0OiBzdHJpbmc7XG5cbiAgLyoqXG4gICAqIFRoZSBsYW5ndWFnZSBvZiB0aGUgc291cmNlIGlmIGB0ZXh0YCBpcyBhIHNvdXJjZSBjb2RlIGZyYWdtZW50LlxuICAgKi9cbiAgcmVhZG9ubHkgbGFuZ3VhZ2U/OiBzdHJpbmc7XG59XG5cbi8qKlxuICogSG92ZXIgaW5mb3JtYXRpb24gZm9yIGEgc3ltYm9sIGF0IHRoZSBob3ZlciBsb2NhdGlvbi5cbiAqL1xuZXhwb3J0IGludGVyZmFjZSBIb3ZlciB7XG4gIC8qKlxuICAgKiBUaGUgaG92ZXIgdGV4dCB0byBkaXNwbGF5IGZvciB0aGUgc3ltYm9sIGF0IHRoZSBob3ZlciBsb2NhdGlvbi4gSWYgdGhlIHRleHQgaW5jbHVkZXNcbiAgICogc291cmNlIGNvZGUsIHRoZSBzZWN0aW9uIHdpbGwgc3BlY2lmeSB3aGljaCBsYW5ndWFnZSBpdCBzaG91bGQgYmUgaW50ZXJwcmV0ZWQgYXMuXG4gICAqL1xuICByZWFkb25seSB0ZXh0OiBIb3ZlclRleHRTZWN0aW9uW107XG5cbiAgLyoqXG4gICAqIFRoZSBzcGFuIG9mIHNvdXJjZSB0aGUgaG92ZXIgY292ZXJzLlxuICAgKi9cbiAgcmVhZG9ubHkgc3BhbjogU3Bhbjtcbn1cblxuLyoqXG4gKiBBbiBpbnN0YW5jZSBvZiBhbiBBbmd1bGFyIGxhbmd1YWdlIHNlcnZpY2UgY3JlYXRlZCBieSBgY3JlYXRlTGFuZ3VhZ2VTZXJ2aWNlKClgLlxuICpcbiAqIFRoZSBsYW5ndWFnZSBzZXJ2aWNlIHJldHVybnMgaW5mb3JtYXRpb24gYWJvdXQgQW5ndWxhciB0ZW1wbGF0ZXMgdGhhdCBhcmUgaW5jbHVkZWQgaW4gYSBwcm9qZWN0XG4gKiBhcyBkZWZpbmVkIGJ5IHRoZSBgTGFuZ3VhZ2VTZXJ2aWNlSG9zdGAuXG4gKlxuICogV2hlbiBhIG1ldGhvZCBleHBlY3RzIGEgYGZpbGVOYW1lYCB0aGlzIGZpbGUgY2FuIGVpdGhlciBiZSBzb3VyY2UgZmlsZSBpbiB0aGUgcHJvamVjdCB0aGF0XG4gKiBjb250YWlucyBhIHRlbXBsYXRlIGluIGEgc3RyaW5nIGxpdGVyYWwgb3IgYSB0ZW1wbGF0ZSBmaWxlIHJlZmVyZW5jZWQgYnkgdGhlIHByb2plY3QgcmV0dXJuZWRcbiAqIGJ5IGBnZXRUZW1wbGF0ZVJlZmVyZW5jZSgpYC4gQWxsIG90aGVyIGZpbGVzIHdpbGwgY2F1c2UgdGhlIG1ldGhvZCB0byByZXR1cm4gYHVuZGVmaW5lZGAuXG4gKlxuICogSWYgYSBtZXRob2QgdGFrZXMgYSBgcG9zaXRpb25gLCBpdCBpcyB0aGUgb2Zmc2V0IG9mIHRoZSBVVEYtMTYgY29kZS1wb2ludCByZWxhdGl2ZSB0byB0aGVcbiAqIGJlZ2lubmluZyBvZiB0aGUgZmlsZSByZWZlcmVuY2UgYnkgYGZpbGVOYW1lYC5cbiAqXG4gKiBUaGlzIGludGVyZmFjZSBhbmQgYWxsIGludGVyZmFjZXMgYW5kIHR5cGVzIG1hcmtlZCBhcyBgTGFuZ3VhZ2VTZXJ2aWNlYCB0eXBlcywgZGVzY3JpYmUgIGFcbiAqIHBhcnRpY3VsYXIgaW1wbGVtZW50YXRpb24gb2YgdGhlIEFuZ3VsYXIgbGFuZ3VhZ2Ugc2VydmljZSBhbmQgaXMgbm90IGludGVuZGVkIHRvIGJlXG4gKiBpbXBsZW1lbnRlZC4gQWRkaW5nIG1lbWJlcnMgdG8gdGhlIGludGVyZmFjZSB3aWxsIG5vdCBiZSBjb25zaWRlcmVkIGEgYnJlYWtpbmcgY2hhbmdlIGFzXG4gKiBkZWZpbmVkIGJ5IFNlbVZlci5cbiAqXG4gKiBSZW1vdmluZyBhIG1lbWJlciBvciBtYWtpbmcgYSBtZW1iZXIgb3B0aW9uYWwsIGNoYW5naW5nIGEgbWV0aG9kIHBhcmFtZXRlcnMsIG9yIGNoYW5naW5nIGFcbiAqIG1lbWJlcidzIHR5cGUgd2lsbCBhbGwgYmUgY29uc2lkZXJlZCBhIGJyZWFraW5nIGNoYW5nZS5cbiAqXG4gKiBXaGlsZSBhbiBpbnRlcmZhY2UgaXMgbWFya2VkIGFzIGV4cGVyaW1lbnRhbCBicmVha2luZy1jaGFuZ2VzIHdpbGwgYmUgYWxsb3dlZCBiZXR3ZWVuIG1pbm9yXG4gKiByZWxlYXNlcy4gQWZ0ZXIgYW4gaW50ZXJmYWNlIGlzIG1hcmtlZCBhcyBzdGFibGUgYnJlYWtpbmctY2hhbmdlcyB3aWxsIG9ubHkgYmUgYWxsb3dlZCBiZXR3ZWVuXG4gKiBtYWpvciByZWxlYXNlcy4gTm8gYnJlYWtpbmcgY2hhbmdlcyBhcmUgYWxsb3dlZCBiZXR3ZWVuIHBhdGNoIHJlbGVhc2VzLlxuICpcbiAqIEBwdWJsaWNBcGlcbiAqL1xuZXhwb3J0IGludGVyZmFjZSBMYW5ndWFnZVNlcnZpY2Uge1xuICAvKipcbiAgICogUmV0dXJucyBhIGxpc3Qgb2YgYWxsIHRoZSBleHRlcm5hbCB0ZW1wbGF0ZXMgcmVmZXJlbmNlZCBieSB0aGUgcHJvamVjdC5cbiAgICovXG4gIGdldFRlbXBsYXRlUmVmZXJlbmNlcygpOiBzdHJpbmdbXXx1bmRlZmluZWQ7XG5cbiAgLyoqXG4gICAqIFJldHVybnMgYSBsaXN0IG9mIGFsbCBlcnJvciBmb3IgYWxsIHRlbXBsYXRlcyBpbiB0aGUgZ2l2ZW4gZmlsZS5cbiAgICovXG4gIGdldERpYWdub3N0aWNzKGZpbGVOYW1lOiBzdHJpbmcpOiBEaWFnbm9zdGljc3x1bmRlZmluZWQ7XG5cbiAgLyoqXG4gICAqIFJldHVybiB0aGUgY29tcGxldGlvbnMgYXQgdGhlIGdpdmVuIHBvc2l0aW9uLlxuICAgKi9cbiAgZ2V0Q29tcGxldGlvbnNBdChmaWxlTmFtZTogc3RyaW5nLCBwb3NpdGlvbjogbnVtYmVyKTogQ29tcGxldGlvbnN8dW5kZWZpbmVkO1xuXG4gIC8qKlxuICAgKiBSZXR1cm4gdGhlIGRlZmluaXRpb24gbG9jYXRpb24gZm9yIHRoZSBzeW1ib2wgYXQgcG9zaXRpb24uXG4gICAqL1xuICBnZXREZWZpbml0aW9uQXQoZmlsZU5hbWU6IHN0cmluZywgcG9zaXRpb246IG51bWJlcik6IERlZmluaXRpb258dW5kZWZpbmVkO1xuXG4gIC8qKlxuICAgKiBSZXR1cm4gdGhlIGhvdmVyIGluZm9ybWF0aW9uIGZvciB0aGUgc3ltYm9sIGF0IHBvc2l0aW9uLlxuICAgKi9cbiAgZ2V0SG92ZXJBdChmaWxlTmFtZTogc3RyaW5nLCBwb3NpdGlvbjogbnVtYmVyKTogSG92ZXJ8dW5kZWZpbmVkO1xuXG4gIC8qKlxuICAgKiBSZXR1cm4gdGhlIHBpcGVzIHRoYXQgYXJlIGF2YWlsYWJsZSBhdCB0aGUgZ2l2ZW4gcG9zaXRpb24uXG4gICAqL1xuICBnZXRQaXBlc0F0KGZpbGVOYW1lOiBzdHJpbmcsIHBvc2l0aW9uOiBudW1iZXIpOiBDb21waWxlUGlwZVN1bW1hcnlbXTtcbn1cbiJdfQ==