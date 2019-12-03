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
        define("@angular/compiler-cli/src/ngtsc/transform/src/api", ["require", "exports"], factory);
    }
})(function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    var HandlerPrecedence;
    (function (HandlerPrecedence) {
        /**
         * Handler with PRIMARY precedence cannot overlap - there can only be one on a given class.
         *
         * If more than one PRIMARY handler matches a class, an error is produced.
         */
        HandlerPrecedence[HandlerPrecedence["PRIMARY"] = 0] = "PRIMARY";
        /**
         * Handlers with SHARED precedence can match any class, possibly in addition to a single PRIMARY
         * handler.
         *
         * It is not an error for a class to have any number of SHARED handlers.
         */
        HandlerPrecedence[HandlerPrecedence["SHARED"] = 1] = "SHARED";
        /**
         * Handlers with WEAK precedence that match a class are ignored if any handlers with stronger
         * precedence match a class.
         */
        HandlerPrecedence[HandlerPrecedence["WEAK"] = 2] = "WEAK";
    })(HandlerPrecedence = exports.HandlerPrecedence || (exports.HandlerPrecedence = {}));
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYXBpLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vLi4vLi4vcGFja2FnZXMvY29tcGlsZXItY2xpL3NyYy9uZ3RzYy90cmFuc2Zvcm0vc3JjL2FwaS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7O0dBTUc7Ozs7Ozs7Ozs7OztJQVVILElBQVksaUJBcUJYO0lBckJELFdBQVksaUJBQWlCO1FBQzNCOzs7O1dBSUc7UUFDSCwrREFBTyxDQUFBO1FBRVA7Ozs7O1dBS0c7UUFDSCw2REFBTSxDQUFBO1FBRU47OztXQUdHO1FBQ0gseURBQUksQ0FBQTtJQUNOLENBQUMsRUFyQlcsaUJBQWlCLEdBQWpCLHlCQUFpQixLQUFqQix5QkFBaUIsUUFxQjVCIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKlxuICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlIGxpY2Vuc2UgdGhhdCBjYW4gYmVcbiAqIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgYXQgaHR0cHM6Ly9hbmd1bGFyLmlvL2xpY2Vuc2VcbiAqL1xuXG5pbXBvcnQge0NvbnN0YW50UG9vbCwgRXhwcmVzc2lvbiwgU3RhdGVtZW50LCBUeXBlfSBmcm9tICdAYW5ndWxhci9jb21waWxlcic7XG5pbXBvcnQgKiBhcyB0cyBmcm9tICd0eXBlc2NyaXB0JztcblxuaW1wb3J0IHtSZWV4cG9ydH0gZnJvbSAnLi4vLi4vaW1wb3J0cyc7XG5pbXBvcnQge0luZGV4aW5nQ29udGV4dH0gZnJvbSAnLi4vLi4vaW5kZXhlcic7XG5pbXBvcnQge0NsYXNzRGVjbGFyYXRpb24sIERlY29yYXRvcn0gZnJvbSAnLi4vLi4vcmVmbGVjdGlvbic7XG5pbXBvcnQge1R5cGVDaGVja0NvbnRleHR9IGZyb20gJy4uLy4uL3R5cGVjaGVjayc7XG5cbmV4cG9ydCBlbnVtIEhhbmRsZXJQcmVjZWRlbmNlIHtcbiAgLyoqXG4gICAqIEhhbmRsZXIgd2l0aCBQUklNQVJZIHByZWNlZGVuY2UgY2Fubm90IG92ZXJsYXAgLSB0aGVyZSBjYW4gb25seSBiZSBvbmUgb24gYSBnaXZlbiBjbGFzcy5cbiAgICpcbiAgICogSWYgbW9yZSB0aGFuIG9uZSBQUklNQVJZIGhhbmRsZXIgbWF0Y2hlcyBhIGNsYXNzLCBhbiBlcnJvciBpcyBwcm9kdWNlZC5cbiAgICovXG4gIFBSSU1BUlksXG5cbiAgLyoqXG4gICAqIEhhbmRsZXJzIHdpdGggU0hBUkVEIHByZWNlZGVuY2UgY2FuIG1hdGNoIGFueSBjbGFzcywgcG9zc2libHkgaW4gYWRkaXRpb24gdG8gYSBzaW5nbGUgUFJJTUFSWVxuICAgKiBoYW5kbGVyLlxuICAgKlxuICAgKiBJdCBpcyBub3QgYW4gZXJyb3IgZm9yIGEgY2xhc3MgdG8gaGF2ZSBhbnkgbnVtYmVyIG9mIFNIQVJFRCBoYW5kbGVycy5cbiAgICovXG4gIFNIQVJFRCxcblxuICAvKipcbiAgICogSGFuZGxlcnMgd2l0aCBXRUFLIHByZWNlZGVuY2UgdGhhdCBtYXRjaCBhIGNsYXNzIGFyZSBpZ25vcmVkIGlmIGFueSBoYW5kbGVycyB3aXRoIHN0cm9uZ2VyXG4gICAqIHByZWNlZGVuY2UgbWF0Y2ggYSBjbGFzcy5cbiAgICovXG4gIFdFQUssXG59XG5cblxuLyoqXG4gKiBQcm92aWRlcyB0aGUgaW50ZXJmYWNlIGJldHdlZW4gYSBkZWNvcmF0b3IgY29tcGlsZXIgZnJvbSBAYW5ndWxhci9jb21waWxlciBhbmQgdGhlIFR5cGVzY3JpcHRcbiAqIGNvbXBpbGVyL3RyYW5zZm9ybS5cbiAqXG4gKiBUaGUgZGVjb3JhdG9yIGNvbXBpbGVycyBpbiBAYW5ndWxhci9jb21waWxlciBkbyBub3QgZGVwZW5kIG9uIFR5cGVzY3JpcHQuIFRoZSBoYW5kbGVyIGlzXG4gKiByZXNwb25zaWJsZSBmb3IgZXh0cmFjdGluZyB0aGUgaW5mb3JtYXRpb24gcmVxdWlyZWQgdG8gcGVyZm9ybSBjb21waWxhdGlvbiBmcm9tIHRoZSBkZWNvcmF0b3JzXG4gKiBhbmQgVHlwZXNjcmlwdCBzb3VyY2UsIGludm9raW5nIHRoZSBkZWNvcmF0b3IgY29tcGlsZXIsIGFuZCByZXR1cm5pbmcgdGhlIHJlc3VsdC5cbiAqL1xuZXhwb3J0IGludGVyZmFjZSBEZWNvcmF0b3JIYW5kbGVyPEEsIE0+IHtcbiAgLyoqXG4gICAqIFRoZSBwcmVjZWRlbmNlIG9mIGEgaGFuZGxlciBjb250cm9scyBob3cgaXQgaW50ZXJhY3RzIHdpdGggb3RoZXIgaGFuZGxlcnMgdGhhdCBtYXRjaCB0aGUgc2FtZVxuICAgKiBjbGFzcy5cbiAgICpcbiAgICogU2VlIHRoZSBkZXNjcmlwdGlvbnMgb24gYEhhbmRsZXJQcmVjZWRlbmNlYCBmb3IgYW4gZXhwbGFuYXRpb24gb2YgdGhlIGJlaGF2aW9ycyBpbnZvbHZlZC5cbiAgICovXG4gIHJlYWRvbmx5IHByZWNlZGVuY2U6IEhhbmRsZXJQcmVjZWRlbmNlO1xuXG4gIC8qKlxuICAgKiBTY2FuIGEgc2V0IG9mIHJlZmxlY3RlZCBkZWNvcmF0b3JzIGFuZCBkZXRlcm1pbmUgaWYgdGhpcyBoYW5kbGVyIGlzIHJlc3BvbnNpYmxlIGZvciBjb21waWxhdGlvblxuICAgKiBvZiBvbmUgb2YgdGhlbS5cbiAgICovXG4gIGRldGVjdChub2RlOiBDbGFzc0RlY2xhcmF0aW9uLCBkZWNvcmF0b3JzOiBEZWNvcmF0b3JbXXxudWxsKTogRGV0ZWN0UmVzdWx0PE0+fHVuZGVmaW5lZDtcblxuXG4gIC8qKlxuICAgKiBBc3luY2hyb25vdXNseSBwZXJmb3JtIHByZS1hbmFseXNpcyBvbiB0aGUgZGVjb3JhdG9yL2NsYXNzIGNvbWJpbmF0aW9uLlxuICAgKlxuICAgKiBgcHJlQW5hbHl6ZWAgaXMgb3B0aW9uYWwgYW5kIGlzIG5vdCBndWFyYW50ZWVkIHRvIGJlIGNhbGxlZCB0aHJvdWdoIGFsbCBjb21waWxhdGlvbiBmbG93cy4gSXRcbiAgICogd2lsbCBvbmx5IGJlIGNhbGxlZCBpZiBhc3luY2hyb25pY2l0eSBpcyBzdXBwb3J0ZWQgaW4gdGhlIENvbXBpbGVySG9zdC5cbiAgICovXG4gIHByZWFuYWx5emU/KG5vZGU6IENsYXNzRGVjbGFyYXRpb24sIG1ldGFkYXRhOiBNKTogUHJvbWlzZTx2b2lkPnx1bmRlZmluZWQ7XG5cbiAgLyoqXG4gICAqIFBlcmZvcm0gYW5hbHlzaXMgb24gdGhlIGRlY29yYXRvci9jbGFzcyBjb21iaW5hdGlvbiwgcHJvZHVjaW5nIGluc3RydWN0aW9ucyBmb3IgY29tcGlsYXRpb25cbiAgICogaWYgc3VjY2Vzc2Z1bCwgb3IgYW4gYXJyYXkgb2YgZGlhZ25vc3RpYyBtZXNzYWdlcyBpZiB0aGUgYW5hbHlzaXMgZmFpbHMgb3IgdGhlIGRlY29yYXRvclxuICAgKiBpc24ndCB2YWxpZC5cbiAgICovXG4gIGFuYWx5emUobm9kZTogQ2xhc3NEZWNsYXJhdGlvbiwgbWV0YWRhdGE6IE0pOiBBbmFseXNpc091dHB1dDxBPjtcblxuICAvKipcbiAgICogUmVnaXN0ZXJzIGluZm9ybWF0aW9uIGFib3V0IHRoZSBkZWNvcmF0b3IgZm9yIHRoZSBpbmRleGluZyBwaGFzZSBpbiBhXG4gICAqIGBJbmRleGluZ0NvbnRleHRgLCB3aGljaCBzdG9yZXMgaW5mb3JtYXRpb24gYWJvdXQgY29tcG9uZW50cyBkaXNjb3ZlcmVkIGluIHRoZVxuICAgKiBwcm9ncmFtLlxuICAgKi9cbiAgaW5kZXg/KGNvbnRleHQ6IEluZGV4aW5nQ29udGV4dCwgbm9kZTogQ2xhc3NEZWNsYXJhdGlvbiwgbWV0YWRhdGE6IEEpOiB2b2lkO1xuXG4gIC8qKlxuICAgKiBQZXJmb3JtIHJlc29sdXRpb24gb24gdGhlIGdpdmVuIGRlY29yYXRvciBhbG9uZyB3aXRoIHRoZSByZXN1bHQgb2YgYW5hbHlzaXMuXG4gICAqXG4gICAqIFRoZSByZXNvbHV0aW9uIHBoYXNlIGhhcHBlbnMgYWZ0ZXIgdGhlIGVudGlyZSBgdHMuUHJvZ3JhbWAgaGFzIGJlZW4gYW5hbHl6ZWQsIGFuZCBnaXZlcyB0aGVcbiAgICogYERlY29yYXRvckhhbmRsZXJgIGEgY2hhbmNlIHRvIGxldmVyYWdlIGluZm9ybWF0aW9uIGZyb20gdGhlIHdob2xlIGNvbXBpbGF0aW9uIHVuaXQgdG8gZW5oYW5jZVxuICAgKiB0aGUgYGFuYWx5c2lzYCBiZWZvcmUgdGhlIGVtaXQgcGhhc2UuXG4gICAqL1xuICByZXNvbHZlPyhub2RlOiBDbGFzc0RlY2xhcmF0aW9uLCBhbmFseXNpczogQSk6IFJlc29sdmVSZXN1bHQ7XG5cbiAgdHlwZUNoZWNrPyhjdHg6IFR5cGVDaGVja0NvbnRleHQsIG5vZGU6IENsYXNzRGVjbGFyYXRpb24sIG1ldGFkYXRhOiBBKTogdm9pZDtcblxuICAvKipcbiAgICogR2VuZXJhdGUgYSBkZXNjcmlwdGlvbiBvZiB0aGUgZmllbGQgd2hpY2ggc2hvdWxkIGJlIGFkZGVkIHRvIHRoZSBjbGFzcywgaW5jbHVkaW5nIGFueVxuICAgKiBpbml0aWFsaXphdGlvbiBjb2RlIHRvIGJlIGdlbmVyYXRlZC5cbiAgICovXG4gIGNvbXBpbGUobm9kZTogQ2xhc3NEZWNsYXJhdGlvbiwgYW5hbHlzaXM6IEEsIGNvbnN0YW50UG9vbDogQ29uc3RhbnRQb29sKTogQ29tcGlsZVJlc3VsdFxuICAgICAgfENvbXBpbGVSZXN1bHRbXTtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBEZXRlY3RSZXN1bHQ8TT4ge1xuICB0cmlnZ2VyOiB0cy5Ob2RlfG51bGw7XG4gIG1ldGFkYXRhOiBNO1xufVxuXG4vKipcbiAqIFRoZSBvdXRwdXQgb2YgYW4gYW5hbHlzaXMgb3BlcmF0aW9uLCBjb25zaXN0aW5nIG9mIHBvc3NpYmx5IGFuIGFyYml0cmFyeSBhbmFseXNpcyBvYmplY3QgKHVzZWQgYXNcbiAqIHRoZSBpbnB1dCB0byBjb2RlIGdlbmVyYXRpb24pIGFuZCBwb3RlbnRpYWxseSBkaWFnbm9zdGljcyBpZiB0aGVyZSB3ZXJlIGVycm9ycyB1bmNvdmVyZWQgZHVyaW5nXG4gKiBhbmFseXNpcy5cbiAqL1xuZXhwb3J0IGludGVyZmFjZSBBbmFseXNpc091dHB1dDxBPiB7XG4gIGFuYWx5c2lzPzogQTtcbiAgZGlhZ25vc3RpY3M/OiB0cy5EaWFnbm9zdGljW107XG4gIGZhY3RvcnlTeW1ib2xOYW1lPzogc3RyaW5nO1xuICB0eXBlQ2hlY2s/OiBib29sZWFuO1xufVxuXG4vKipcbiAqIEEgZGVzY3JpcHRpb24gb2YgdGhlIHN0YXRpYyBmaWVsZCB0byBhZGQgdG8gYSBjbGFzcywgaW5jbHVkaW5nIGFuIGluaXRpYWxpemF0aW9uIGV4cHJlc3Npb25cbiAqIGFuZCBhIHR5cGUgZm9yIHRoZSAuZC50cyBmaWxlLlxuICovXG5leHBvcnQgaW50ZXJmYWNlIENvbXBpbGVSZXN1bHQge1xuICBuYW1lOiBzdHJpbmc7XG4gIGluaXRpYWxpemVyOiBFeHByZXNzaW9uO1xuICBzdGF0ZW1lbnRzOiBTdGF0ZW1lbnRbXTtcbiAgdHlwZTogVHlwZTtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBSZXNvbHZlUmVzdWx0IHtcbiAgcmVleHBvcnRzPzogUmVleHBvcnRbXTtcbiAgZGlhZ25vc3RpY3M/OiB0cy5EaWFnbm9zdGljW107XG59XG4iXX0=